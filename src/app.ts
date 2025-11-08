import * as tf from '@tensorflow/tfjs'

type AudioProfileMode = 'high-precision' | 'medium-precision' | 'low-precision'
type DrawProgress = {
  percent: number
  etaMs: number | null
}

let statusNode = querySelector('#status')
let input = querySelector<HTMLInputElement>('#fileInput')
let profileSelect = querySelector<HTMLSelectElement>('#profileSelect')
let canvas = querySelector<HTMLCanvasElement>('#canvas')
let statsNode = querySelector('#stats')
let maxFrequencyInput = querySelector<HTMLInputElement>('#maxFrequencyInput')

let setStatus = (message: string) => {
  statusNode.textContent = message
}

let setStats = (html: string) => {
  statsNode.innerHTML = html
}

let applyProfileDefaults = () => {
  let mode = profileSelect.value as AudioProfileMode
  let profile = getAudioProfile(mode)
  maxFrequencyInput.value = String(profile.maxFrequency)
}

let renderStats = (inputs: Record<string, string | number>) => {
  setStats(
    Object.entries(inputs)
      .map(([key, value]) => `<pre>${key}: ${value}</pre>`)
      .join(''),
  )
}

let describeProgress = (progress: DrawProgress) => {
  let etaText = formatEta(progress.etaMs)
  return `Drawing spectrogram ${progress.percent}% (${etaText})`
}

let getNow = () => performance.now()

function formatEta(etaMs: number | null) {
  if (etaMs == null || !isFinite(etaMs) || etaMs < 0) {
    return 'ETA --'
  }
  let seconds = etaMs / 1000
  if (seconds >= 60) {
    let minutes = Math.floor(seconds / 60)
    let remainingSeconds = Math.round(seconds - minutes * 60)
    return `ETA ${minutes}m ${remainingSeconds}s`
  }
  let roundedSeconds = Math.max(0, Math.round(seconds * 10) / 10)
  return `ETA ${roundedSeconds}s`
}

export async function loadFile(file: File) {
  let arrayBuffer = await file.arrayBuffer()
  return arrayBuffer
}

export async function loadUrl(url: string) {
  let res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to load file from ${url}, status: ${res.status}`)
  }
  let arrayBuffer = await res.arrayBuffer()
  return arrayBuffer
}

export async function decodeAudio(
  arrayBuffer: ArrayBuffer,
  sampleRate?: number,
) {
  let audioContext = new AudioContext({ latencyHint: 'playback', sampleRate })
  let audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
  return audioBuffer
}

export function getMonoAudioData(audioBuffer: AudioBuffer) {
  let audioData = audioBuffer.getChannelData(0)
  let channels = audioBuffer.numberOfChannels
  if (channels == 1) {
    return audioData
  }
  for (let c = 1; c < channels; c++) {
    let channelData = audioBuffer.getChannelData(c)
    for (let i = 0; i < audioData.length; i++) {
      audioData[i] += channelData[i]
    }
  }
  for (let i = 0; i < audioData.length; i++) {
    audioData[i] /= channels
  }
  return audioData
}

export function getSpectrogramData(
  audioData: Float32Array,
  windowSize: number,
) {
  let stride = windowSize / 2
  let frameCount = Math.floor((audioData.length - windowSize) / stride) + 1
  let frequencyBinCount = windowSize / 2

  let spectrogramData: number[][] = []

  tf.tidy(() => {
    let hammingWindow = tf.signal.hammingWindow(windowSize)
    for (let frame = 0; frame < frameCount; frame++) {
      let start = frame * stride
      let end = start + windowSize
      let frameBuffer = audioData.slice(start, end)

      let windowedFrame = tf.mul(tf.tensor1d(frameBuffer), hammingWindow)

      let complexFrame = tf.complex(windowedFrame, tf.zerosLike(windowedFrame))

      let fft = tf.spectral.fft(complexFrame)

      let magnitude = tf.abs(fft).slice([0], [windowSize / 2])

      let logMagnitude = tf.log1p(magnitude)

      spectrogramData.push(logMagnitude.arraySync() as number[])
    }
  })

  return spectrogramData
}

export async function drawSpectrogram(inputs: {
  signal: AbortSignal
  audioData: Float32Array
  windowSize: number
  hopSize: number
  maxFrequency: number
  canvas: HTMLCanvasElement
  onProgress?: (progress: DrawProgress) => void
}) {
  let {
    signal,
    audioData,
    windowSize,
    hopSize,
    maxFrequency,
    canvas,
    onProgress,
  } = inputs

  let frameCount = Math.floor((audioData.length - windowSize) / hopSize) + 1
  let frequencyBinCount = windowSize / 2

  if (maxFrequency > frequencyBinCount) {
    throw new Error(
      `maxFrequency (${maxFrequency}) must be less than frequencyBinCount (${frequencyBinCount})`,
    )
  }

  {
    let size = frameCount * maxFrequency
    let ans = confirm(
      `canvas size: ${frameCount}x${maxFrequency} (required: ${size.toLocaleString()})`,
    )
    if (!ans) {
      return
    }
  }

  canvas.width = frameCount
  canvas.height = maxFrequency

  let context = canvas.getContext('2d')!

  let imageData = context.createImageData(frameCount, maxFrequency)

  let hammingWindow = tf.signal.hammingWindow(windowSize)
  let timer = setInterval(draw)
  let lastPercent = -1
  let startTime = getNow()
  for (let x = 0; x < frameCount && !signal.aborted; x++) {
    let frame = Math.floor((x / canvas.width) * frameCount)
    let start = frame * hopSize
    let end = start + windowSize
    let frameBuffer = audioData.slice(start, end)

    let windowedFrame = tf.mul(tf.tensor1d(frameBuffer), hammingWindow)

    let logMagnitude = tf.tidy(() => {
      let complexFrame = tf.complex(windowedFrame, tf.zerosLike(windowedFrame))

      let fft = tf.spectral.fft(complexFrame)

      let magnitude = tf.abs(fft.slice([0], [windowSize / 2]))

      let logMagnitude = tf.div(tf.log1p(magnitude), log_255)

      return logMagnitude
    })

    let values = (await logMagnitude.data()) as Float32Array
    logMagnitude.dispose()

    for (let freq = 0; freq < maxFrequency; freq++) {
      let y = maxFrequency - freq - 1
      let mag = values[freq]
      let color = Math.floor(mag * 255)
      let index = (y * canvas.width + x) * 4
      imageData.data[index] = color
      imageData.data[index + 1] = color
      imageData.data[index + 2] = color
      imageData.data[index + 3] = 255
    }
    if (onProgress) {
      let percent = Math.floor(((x + 1) / frameCount) * 100)
      if (percent !== lastPercent) {
        lastPercent = percent
        let progress = (x + 1) / frameCount
        let elapsedMs = getNow() - startTime
        let speed = progress / elapsedMs
        let etaMs = (1 - progress) / speed
        onProgress({ percent, etaMs })
      }
    }
  }
  clearInterval(timer)

  if (onProgress && !signal.aborted) {
    onProgress({ percent: 100, etaMs: 0 })
  }

  function draw() {
    context.putImageData(imageData, 0, 0)
  }

  draw()
}

const log_255 = Math.log1p(255)

let abortController = new AbortController()
export async function run() {
  try {
    abortController.abort()
    abortController = new AbortController()
    await main(abortController.signal)
  } catch (error) {
    console.error(error)
    alert(String(error))
    setStatus('Failed to render spectrogram')
  }
}

export async function main(signal: AbortSignal) {
  let file = input.files?.[0]
  if (!file) {
    setStatus('Select an audio file to render')
    setStats('')
    return
  }

  setStatus('Loading file...')
  let mode = profileSelect.value as AudioProfileMode
  let profile = getAudioProfile(mode)
  let {
    sampleRate,
    windowSize,
    hopSize,
    maxFrequency: defaultMaxFrequency,
  } = profile
  let maxFrequency = (() => {
    let raw = maxFrequencyInput.value.trim()
    if (!raw) {
      return defaultMaxFrequency
    }
    let parsed = Number(raw)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error('Max frequency must be a positive number')
    }
    return Math.floor(parsed)
  })()
  let maxAllowedFrequency = Math.floor(windowSize / 2)
  if (maxFrequency > maxAllowedFrequency) {
    throw new Error(
      `Max frequency (${maxFrequency}) must be <= ${maxAllowedFrequency} for windowSize ${windowSize}`,
    )
  }

  console.log('file size:', file.size.toLocaleString())

  console.time('loadFile')
  let arrayBuffer = await loadFile(file)
  console.timeEnd('loadFile')
  setStatus('Decoding audio...')

  console.time('decodeAudio')
  let audioBuffer = await decodeAudio(arrayBuffer, sampleRate)
  console.timeEnd('decodeAudio')
  setStatus('Preparing audio data...')

  console.log('audio duration:', audioBuffer.duration)

  console.time('getMonoAudioData')
  let audioData = getMonoAudioData(audioBuffer)
  console.timeEnd('getMonoAudioData')
  setStatus(describeProgress({ percent: 0, etaMs: null }))
  let frameCount = Math.floor((audioData.length - windowSize) / hopSize) + 1
  renderStats({
    filename: file.name,
    file_size: file.size,
    duration: audioBuffer.duration.toFixed(2) + 's',
    sample_rate: sampleRate + ' Hz',
    window_size: windowSize,
    hop_size: hopSize,
    max_frequency: maxFrequency + ' Hz',
    frame_count: frameCount,
  })

  console.time('drawSpectrogram')
  await drawSpectrogram({
    signal,
    audioData,
    windowSize,
    hopSize,
    maxFrequency,
    canvas,
    onProgress: progress => {
      setStatus(describeProgress(progress))
    },
  })
  console.timeEnd('drawSpectrogram')

  console.log('spectrogram size:', canvas.width + 'x' + canvas.height)
  if (!signal.aborted) {
    setStatus('Spectrogram ready')
  }
}

function getAudioProfile(mode: AudioProfileMode) {
  let k = 1000
  if (mode === 'high-precision') {
    return {
      sampleRate: 44.1 * k,
      windowSize: 8192,
      hopSize: 512,
      maxFrequency: 1000,
    }
  }
  if (mode === 'medium-precision') {
    return {
      sampleRate: 32 * k,
      windowSize: 4096,
      hopSize: 512,
      maxFrequency: 1000,
    }
  }
  if (mode === 'low-precision') {
    return {
      sampleRate: 16 * k,
      windowSize: 2048,
      hopSize: 256,
      maxFrequency: 1024,
    }
  }
  throw new Error(`Unsupported mode: ${mode}`)
}

setStatus('Ready')
setStats('')
applyProfileDefaults()

input.onchange = run
profileSelect.onchange = () => {
  applyProfileDefaults()
  run()
}
maxFrequencyInput.onchange = run
run()

function querySelector<E extends HTMLElement>(selector: string) {
  let element = document.querySelector<E>(selector)
  if (!element) {
    throw new Error(`Element not found: "${selector}"`)
  }
  return element
}
