import type { AudioProfileMode, DrawProgress } from './core'
import {
  decodeAudio,
  drawSpectrogram,
  getAudioProfile,
  getMonoAudioData,
  getNow,
  loadFile,
} from './core'

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
