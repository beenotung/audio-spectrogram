import * as tf from '@tensorflow/tfjs'

export type AudioProfileMode =
  | 'high-precision'
  | 'medium-precision'
  | 'low-precision'
export type DrawProgress = {
  percent: number
  etaMs: number | null
}

export function getNow() {
  return performance.now()
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

export type PauseState = {
  paused: boolean
}

export async function drawSpectrogram(inputs: {
  signal: AbortSignal
  audioData: Float32Array
  windowSize: number
  hopSize: number
  maxFrequency: number
  canvas: HTMLCanvasElement
  onProgress?: (progress: DrawProgress) => void
  frameStart?: number
  frameEnd?: number
  pauseState?: PauseState
}) {
  let {
    signal,
    audioData,
    windowSize,
    hopSize,
    maxFrequency,
    canvas,
    onProgress,
    frameStart,
    frameEnd,
    pauseState,
  } = inputs

  let totalFrameCount =
    Math.floor((audioData.length - windowSize) / hopSize) + 1
  let frequencyBinCount = windowSize / 2

  if (maxFrequency > frequencyBinCount) {
    throw new Error(
      `maxFrequency (${maxFrequency}) must be less than frequencyBinCount (${frequencyBinCount})`,
    )
  }

  // Use existing canvas size - don't resize
  let canvasWidth = canvas.width
  let canvasHeight = canvas.height

  if (canvasWidth <= 0 || canvasHeight <= 0) {
    throw new Error('Canvas width and height must be greater than 0')
  }

  // Determine frame range to render (default to all frames)
  let startFrame = frameStart ?? 0
  let endFrame = frameEnd ?? totalFrameCount
  startFrame = Math.max(0, Math.min(startFrame, totalFrameCount - 1))
  endFrame = Math.max(0, Math.min(endFrame, totalFrameCount))
  let frameCount = endFrame - startFrame

  if (frameCount <= 0) {
    throw new Error(
      'Invalid frame range: frameStart must be less than frameEnd',
    )
  }

  let context = canvas.getContext('2d')!

  let imageData = context.createImageData(canvasWidth, canvasHeight)

  let hammingWindow = tf.signal.hammingWindow(windowSize)
  let timer: ReturnType<typeof setInterval> | null = setInterval(draw)
  let lastPercent = -1
  let startTime = getNow()
  let frameMagnitudes: tf.Tensor[] = []

  // Helper function to wait for resume if paused
  let waitForResume = async () => {
    if (!pauseState) return
    while (pauseState.paused && !signal.aborted) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }
  }

  // Cleanup function to dispose all tensors
  let cleanup = () => {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
    // Dispose hamming window
    if (hammingWindow) {
      hammingWindow.dispose()
      hammingWindow = null as any
    }
    // Dispose any remaining frame magnitude tensors
    for (let tensor of frameMagnitudes) {
      if (tensor && !tensor.isDisposed) {
        tensor.dispose()
      }
    }
    frameMagnitudes = []
  }

  function draw() {
    context.putImageData(imageData, 0, 0)
  }

  try {
    // Render each pixel column
    for (let x = 0; x < canvasWidth && !signal.aborted; x++) {
      // Check for pause and wait if paused
      await waitForResume()
      if (signal.aborted) break
      // Map canvas x position to frame index range
      // x=0 maps to startFrame, x=canvasWidth-1 maps to endFrame-1
      let frameIndexStart = startFrame + (x / canvasWidth) * frameCount
      let frameIndexEnd = startFrame + ((x + 1) / canvasWidth) * frameCount

      // Determine which frames contribute to this pixel (max pooling)
      let frameStartIdx = Math.floor(frameIndexStart)
      let frameEndIdx = Math.ceil(frameIndexEnd)

      // Clamp frame range to valid values
      frameStartIdx = Math.max(
        startFrame,
        Math.min(frameStartIdx, startFrame + frameCount - 1),
      )
      frameEndIdx = Math.max(
        startFrame,
        Math.min(frameEndIdx, startFrame + frameCount),
      )

      // Process all frames that map to this pixel (max pooling horizontally)
      // Use TensorFlow.js to compute max across frames to avoid data transfer overhead
      // Clear previous frame magnitudes
      for (let tensor of frameMagnitudes) {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose()
        }
      }
      frameMagnitudes = []

      for (
        let frame = frameStartIdx;
        frame < frameEndIdx && !signal.aborted;
        frame++
      ) {
        let start = frame * hopSize
        let end = start + windowSize
        if (end > audioData.length) {
          end = audioData.length
          start = end - windowSize
        }
        if (start < 0) {
          start = 0
          end = Math.min(windowSize, audioData.length)
        }
        let frameBuffer = audioData.slice(start, end)

        let windowedFrame = tf.mul(tf.tensor1d(frameBuffer), hammingWindow)

        let logMagnitude = tf.tidy(() => {
          let complexFrame = tf.complex(
            windowedFrame,
            tf.zerosLike(windowedFrame),
          )

          let fft = tf.spectral.fft(complexFrame)

          let magnitude = tf.abs(fft.slice([0], [windowSize / 2]))

          let logMagnitude = tf.div(tf.log1p(magnitude), log_255)

          return logMagnitude
        })

        // Dispose windowedFrame after tidy (it's no longer needed)
        windowedFrame.dispose()

        // Keep in TensorFlow.js - slice to maxFrequency
        let sliced = logMagnitude.slice([0], [maxFrequency])
        logMagnitude.dispose()
        frameMagnitudes.push(sliced)
      }

      // Compute max across all frames using TensorFlow.js (faster, stays on GPU if available)
      let maxValuesTensor: tf.Tensor
      if (frameMagnitudes.length > 0) {
        if (frameMagnitudes.length === 1) {
          maxValuesTensor = frameMagnitudes[0]
        } else {
          // Stack frames and take max along the batch dimension
          let stacked = tf.stack(frameMagnitudes)
          maxValuesTensor = tf.max(stacked, 0)
          stacked.dispose()
          // Dispose individual frame tensors
          for (let tensor of frameMagnitudes) {
            tensor.dispose()
          }
        }
        // Get data only once after all TensorFlow.js operations
        let maxValues = (await maxValuesTensor.data()) as Float32Array
        maxValuesTensor.dispose()

        // Write the maximum values to the pixel column, with vertical max pooling
        for (let y = 0; y < canvasHeight; y++) {
          // Map canvas y position to frequency bin range
          // y=0 is top (highest frequency), y=canvasHeight-1 is bottom (lowest frequency)
          // Frequency bins are indexed from 0 (lowest) to maxFrequency-1 (highest)
          // So we need to invert: canvas y maps to frequency bin (maxFrequency - 1 - ...)
          let freqIndexStart =
            ((canvasHeight - 1 - y) / canvasHeight) * maxFrequency
          let freqIndexEnd = ((canvasHeight - y) / canvasHeight) * maxFrequency

          // Determine which frequency bins contribute to this pixel (max pooling vertically)
          let freqStart = Math.floor(freqIndexStart)
          let freqEnd = Math.ceil(freqIndexEnd)

          // Clamp frequency range to valid values
          freqStart = Math.max(0, Math.min(freqStart, maxFrequency - 1))
          freqEnd = Math.max(0, Math.min(freqEnd, maxFrequency))

          // Find maximum value across frequency bins for this pixel
          let maxMag = 0
          for (let freq = freqStart; freq < freqEnd; freq++) {
            if (maxValues[freq] > maxMag) {
              maxMag = maxValues[freq]
            }
          }

          let color = Math.floor(maxMag * 255)
          let index = (y * canvasWidth + x) * 4
          imageData.data[index] = color
          imageData.data[index + 1] = color
          imageData.data[index + 2] = color
          imageData.data[index + 3] = 255
        }
      } else {
        // No frames to process - fill with black
        for (let y = 0; y < canvasHeight; y++) {
          let index = (y * canvasWidth + x) * 4
          imageData.data[index] = 0
          imageData.data[index + 1] = 0
          imageData.data[index + 2] = 0
          imageData.data[index + 3] = 255
        }
      }
      if (onProgress) {
        let percent = Math.floor(((x + 1) / canvasWidth) * 100)
        if (percent !== lastPercent) {
          lastPercent = percent
          let progress = (x + 1) / canvasWidth
          let elapsedMs = getNow() - startTime
          let speed = progress / elapsedMs
          let etaMs = (1 - progress) / speed
          onProgress({ percent, etaMs })
        }
      }
    }
  } finally {
    // Always cleanup, even if aborted
    // Clean up any remaining frame magnitude tensors from last iteration
    for (let tensor of frameMagnitudes) {
      if (tensor && !tensor.isDisposed) {
        tensor.dispose()
      }
    }
    frameMagnitudes = []
    cleanup()
  }

  if (onProgress && !signal.aborted) {
    onProgress({ percent: 100, etaMs: 0 })
  }

  draw()
}

const log_255 = Math.log1p(255)

export function calculateDefaultZoom(
  frameCount: number,
  viewportWidth: number,
): number {
  // Calculate zoom to fit entire audio in viewport
  // zoom = viewportWidth / frameCount
  // This means 1 pixel per frame when zoom = 1, but we want to fit all frames
  if (frameCount <= 0) return 1
  return viewportWidth / frameCount
}

export function getAudioProfile(mode: AudioProfileMode) {
  let k = 1000
  if (mode === 'high-precision') {
    return {
      sampleRate: 44.1 * k,
      windowSize: 8192,
      hopSize: 512,
      maxFrequency: 1000, // bins, will be converted to Hz
    }
  }
  if (mode === 'medium-precision') {
    return {
      sampleRate: 32 * k,
      windowSize: 4096,
      hopSize: 512,
      maxFrequency: 1000, // bins, will be converted to Hz
    }
  }
  if (mode === 'low-precision') {
    return {
      sampleRate: 16 * k,
      windowSize: 2048,
      hopSize: 256,
      maxFrequency: 1024, // bins, will be converted to Hz
    }
  }
  throw new Error(`Unsupported mode: ${mode}`)
}

// Default max frequency in Hz for speech range (telephone quality: 0-3400 Hz, high quality: 0-8000 Hz)
// Using 3500 Hz as a good default for speech analysis
export const DEFAULT_MAX_FREQUENCY_HZ = 3500
// Default canvas height in pixels
export const DEFAULT_MAX_HEIGHT_PX = 250

// Calculate waveform data (downsampled for fast rendering)
export function calculateWaveformData(
  audioData: Float32Array,
  canvasWidth: number,
) {
  let samplesPerPixel = Math.max(1, Math.floor(audioData.length / canvasWidth))
  let waveformData: { min: number; max: number; rms: number }[] = []

  for (let x = 0; x < canvasWidth; x++) {
    let start = x * samplesPerPixel
    let end = Math.min(start + samplesPerPixel, audioData.length)
    if (start >= audioData.length) {
      waveformData.push({ min: 0, max: 0, rms: 0 })
      continue
    }

    let min = audioData[start]
    let max = audioData[start]
    let sumSquares = 0
    let count = 0

    for (let i = start; i < end; i++) {
      let value = audioData[i]
      if (value < min) min = value
      if (value > max) max = value
      sumSquares += value * value
      count++
    }

    let rms = Math.sqrt(sumSquares / count)
    waveformData.push({ min, max, rms })
  }

  return waveformData
}

// Draw waveform on canvas
export function drawWaveform(
  canvas: HTMLCanvasElement,
  waveformData: { min: number; max: number; rms: number }[],
) {
  let context = canvas.getContext('2d')!
  let canvasWidth = canvas.width
  let canvasHeight = canvas.height
  let centerY = canvasHeight / 2

  // Clear canvas
  context.fillStyle = '#000'
  context.fillRect(0, 0, canvasWidth, canvasHeight)

  // Draw waveform
  context.strokeStyle = '#0f0'
  context.fillStyle = '#0f0'
  context.lineWidth = 1

  // Draw full waveform
  for (let i = 0; i < waveformData.length && i < canvasWidth; i++) {
    let data = waveformData[i]
    let x = i

    // Draw RMS as main line (centered)
    let rmsHeight = (data.rms * centerY) / 1.0 // Scale to fit
    context.beginPath()
    context.moveTo(x, centerY - rmsHeight)
    context.lineTo(x, centerY + rmsHeight)
    context.stroke()

    // Draw min/max as thin lines
    let minY = centerY - (data.min * centerY) / 1.0
    let maxY = centerY - (data.max * centerY) / 1.0
    context.beginPath()
    context.moveTo(x, minY)
    context.lineTo(x, maxY)
    context.stroke()
  }
}
