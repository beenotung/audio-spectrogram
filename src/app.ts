import type { AudioProfileMode, DrawProgress, PauseState } from './core'
import {
  calculateDefaultZoom,
  calculateWaveformData,
  DEFAULT_MAX_FREQUENCY_HZ,
  DEFAULT_MAX_HEIGHT_PX,
  decodeAudio,
  drawSpectrogram,
  drawWaveform,
  getAudioProfile,
  getMonoAudioData,
  getNow,
  loadFile,
} from './core'

let statusNode = querySelector('#status')
let input = querySelector<HTMLInputElement>('#fileInput')
let profileSelect = querySelector<HTMLSelectElement>('#profileSelect')
let waveformCanvas = querySelector<HTMLCanvasElement>('#waveformCanvas')
let waveformOverlayCanvas = querySelector<HTMLCanvasElement>(
  '#waveformOverlayCanvas',
)
let canvas = querySelector<HTMLCanvasElement>('#canvas')
let canvasContainer = querySelector('#canvasContainer')
let statsNode = querySelector('#stats')
let cursorInfoNode = querySelector('#cursorInfo')
let maxFrequencyInput = querySelector<HTMLInputElement>('#maxFrequencyInput')
let maxHeightInput = querySelector<HTMLInputElement>('#maxHeightInput')
let renderSpectrogramBtn = querySelector<HTMLButtonElement>(
  '#renderSpectrogramBtn',
)
let pauseBtn = querySelector<HTMLButtonElement>('#pauseBtn')
let resumeBtn = querySelector<HTMLButtonElement>('#resumeBtn')
let zoomSlider = querySelector<HTMLInputElement>('#zoomSlider')
let zoomInput = querySelector<HTMLInputElement>('#zoomInput')
let zoomOutBtn = querySelector<HTMLButtonElement>('#zoomOutBtn')
let zoomInBtn = querySelector<HTMLButtonElement>('#zoomInBtn')
let panLeftBtn = querySelector<HTMLButtonElement>('#panLeftBtn')
let panRightBtn = querySelector<HTMLButtonElement>('#panRightBtn')

let zoom = 1 // Internal zoom value (for backward compatibility)
let zoomSeconds = 1 // Zoom in seconds (what user sees)
let offset = 0
let frameCount = 0
let maxHeight = DEFAULT_MAX_HEIGHT_PX
let audioDuration = 0 // Total audio duration in seconds
let cachedAudioData: Float32Array | null = null
let cachedParams: {
  windowSize: number
  hopSize: number
  maxFrequency: number
  sampleRate: number
} | null = null
let cachedWaveformData: { min: number; max: number; rms: number }[] | null =
  null
let hoverTimestamp: number | null = null // Track hover position from spectrogram
// Store the frame range that was rendered on the canvas
let renderedFrameStart = 0 // First frame index rendered on canvas
let renderedFrameEnd = 0 // Last frame index rendered on canvas (exclusive)

let setStatus = (message: string) => {
  statusNode.textContent = message
}

let setStats = (html: string) => {
  statsNode.innerHTML = html
}

let applyProfileDefaults = () => {
  // Only set defaults if inputs are empty (to respect Firefox's preserved values)
  if (!maxFrequencyInput.value.trim()) {
    maxFrequencyInput.value = String(DEFAULT_MAX_FREQUENCY_HZ)
  }

  if (!maxHeightInput.value.trim()) {
    maxHeightInput.value = String(DEFAULT_MAX_HEIGHT_PX)
  }
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

function formatTime(seconds: number): string {
  // Format as mm:ss.sss
  let minutes = Math.floor(seconds / 60)
  let secs = seconds % 60
  let secsStr = secs.toFixed(3).padStart(6, '0')
  return `${minutes}:${secsStr}`
}

function formatTimeBoth(seconds: number): string {
  // Return both formats: "mm:ss.sss (X.XXXs)"
  return `${formatTime(seconds)} (${seconds.toFixed(3)}s)`
}

let abortController = new AbortController()
let pauseState: PauseState = { paused: false }

let updatePauseButtonVisibility = () => {
  if (pauseState.paused) {
    pauseBtn.style.display = 'none'
    resumeBtn.style.display = ''
  } else {
    pauseBtn.style.display = ''
    resumeBtn.style.display = 'none'
  }
}

export async function run() {
  try {
    abortController.abort()
    abortController = new AbortController()
    pauseState.paused = false
    updatePauseButtonVisibility()
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
  // maxFrequency input is in Hz, convert to number of bins
  let maxFrequencyHz = (() => {
    let raw = maxFrequencyInput.value.trim()
    if (!raw) {
      // Use default for speech range if not set
      return DEFAULT_MAX_FREQUENCY_HZ
    }
    let parsed = Number(raw)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new Error('Max frequency must be a positive number')
    }
    return parsed
  })()

  // Convert Hz to number of bins
  let binWidth = sampleRate / windowSize
  let maxFrequency = Math.floor(maxFrequencyHz / binWidth)

  // Clamp to available bins
  let maxAllowedBins = Math.floor(windowSize / 2)
  if (maxFrequency > maxAllowedBins) {
    maxFrequency = maxAllowedBins
  }
  if (maxFrequency < 1) {
    maxFrequency = 1
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

  // Cache audio data and params for zoom/pan
  cachedAudioData = audioData
  cachedParams = { windowSize, hopSize, maxFrequency, sampleRate }

  // Calculate and draw waveform (fast preview)
  setStatus('Calculating waveform preview...')
  let containerWidth = canvasContainer.clientWidth || 1920
  // Get device pixel ratio for crisp rendering
  let dpr = window.devicePixelRatio || 1
  waveformCanvas.width = containerWidth * dpr
  waveformCanvas.height = 100 * dpr
  // Set overlay canvas to same dimensions (matching resolution)
  waveformOverlayCanvas.width = containerWidth * dpr
  waveformOverlayCanvas.height = 100 * dpr
  // Scale context to account for device pixel ratio
  let waveformContext = waveformCanvas.getContext('2d')!
  waveformContext.scale(dpr, dpr)
  let overlayContext = waveformOverlayCanvas.getContext('2d')!
  overlayContext.scale(dpr, dpr)
  cachedWaveformData = calculateWaveformData(audioData, containerWidth)
  drawWaveform(waveformCanvas, cachedWaveformData)
  setStatus('Waveform ready')

  setStatus(describeProgress({ percent: 0, etaMs: null }))
  frameCount = Math.floor((audioData.length - windowSize) / hopSize) + 1

  // Store audio duration
  audioDuration = audioBuffer.duration

  // Calculate default zoom to fit entire audio
  let viewportWidth = canvasContainer.clientWidth || 1920
  zoom = calculateDefaultZoom(frameCount, viewportWidth)
  // Convert zoom to seconds
  zoomSeconds = zoomToSeconds(zoom)
  offset = 0
  updateZoomDisplay()
  updateZoomSliderRange()

  // Get max height from input, use default if not set
  let maxHeightValue = maxHeightInput.value.trim()
  if (maxHeightValue) {
    let parsed = Number(maxHeightValue)
    if (Number.isFinite(parsed) && parsed >= 100) {
      maxHeight = parsed
    } else {
      maxHeight = DEFAULT_MAX_HEIGHT_PX
    }
  } else {
    maxHeight = DEFAULT_MAX_HEIGHT_PX
  }

  renderStats({
    filename: file.name,
    file_size: file.size,
    duration: formatTimeBoth(audioBuffer.duration),
    sample_rate: sampleRate + ' Hz',
    window_size: windowSize,
    hop_size: hopSize,
    max_frequency: maxFrequency + ' Hz',
    frame_count: frameCount,
  })

  setStatus('Waveform ready - Click "Render Spectrogram" to generate')
  // Don't render spectrogram automatically - wait for user to click button
}

async function renderSpectrogram(
  signal: AbortSignal,
  audioData: Float32Array,
  windowSize: number,
  hopSize: number,
  maxFrequency: number,
) {
  // Get max height from input (fixed height), use default if not set
  let maxHeightValue = maxHeightInput.value.trim()
  if (maxHeightValue) {
    let parsed = Number(maxHeightValue)
    if (Number.isFinite(parsed) && parsed >= 100) {
      maxHeight = parsed
    } else {
      maxHeight = DEFAULT_MAX_HEIGHT_PX
    }
  } else {
    maxHeight = DEFAULT_MAX_HEIGHT_PX
  }
  let canvasHeight = maxHeight

  // Use maximum available width from parent container
  // The canvas CSS is set to width: 100% to fill the container
  // We need to get the actual pixel width of the container
  let containerWidth = canvasContainer.clientWidth || 1920

  // Set canvas internal dimensions
  // Width uses full container width, height uses fixed value from input
  canvas.width = containerWidth
  canvas.height = canvasHeight

  // Calculate frame range based on time window and offset
  // Always limit rendering to the visible time window to improve performance
  let frameStart: number | undefined
  let frameEnd: number | undefined

  // Calculate time window in seconds
  let timeWindowSeconds = zoomSeconds
  let showsFullAudio = timeWindowSeconds >= audioDuration

  if (!showsFullAudio) {
    // Calculate which frames correspond to the visible time window
    // Convert time window and offset to frame indices
    let totalDuration = audioDuration
    let startTimeRatio = offset / (frameCount * zoom) // Ratio of total canvas
    let startTimeSeconds = startTimeRatio * totalDuration
    let endTimeSeconds = startTimeSeconds + timeWindowSeconds

    // Convert time to frame indices
    // frame = (time * sampleRate) / hopSize
    let startFrameIndex = Math.floor(
      (startTimeSeconds * cachedParams!.sampleRate) / cachedParams!.hopSize,
    )
    let endFrameIndex = Math.ceil(
      (endTimeSeconds * cachedParams!.sampleRate) / cachedParams!.hopSize,
    )

    // Clamp to valid frame range
    frameStart = Math.max(0, Math.min(startFrameIndex, frameCount - 1))
    frameEnd = Math.max(0, Math.min(endFrameIndex, frameCount))

    // Ensure we have at least some frames to render
    if (frameEnd <= frameStart) {
      frameEnd = Math.min(frameStart + 1, frameCount)
    }
  }
  // If time window >= audio duration, render all frames (undefined frameStart/frameEnd)

  // Store the frame range that will be rendered
  renderedFrameStart = frameStart ?? 0
  renderedFrameEnd = frameEnd ?? frameCount

  console.time('drawSpectrogram')
  await drawSpectrogram({
    signal,
    audioData,
    windowSize,
    hopSize,
    maxFrequency,
    canvas,
    frameStart,
    frameEnd,
    pauseState,
    onProgress: progress => {
      setStatus(describeProgress(progress))
    },
  })
  console.timeEnd('drawSpectrogram')

  // Update scroll position based on offset
  updateScrollPosition()

  // Update waveform viewport indicator
  updateWaveformViewport()

  console.log('spectrogram size:', canvas.width + 'x' + canvas.height)
  if (!signal.aborted) {
    setStatus('Spectrogram ready')
  }
}

setStatus('Ready')
setStats('')
applyProfileDefaults()

let updateScrollPosition = () => {
  canvasContainer.scrollLeft = offset
}

let updateWaveformViewport = () => {
  if (!cachedWaveformData || !cachedParams || frameCount === 0) return

  let context = waveformCanvas.getContext('2d')!
  let canvasWidth = waveformCanvas.width
  let canvasHeight = waveformCanvas.height

  // Redraw waveform
  drawWaveform(waveformCanvas, cachedWaveformData)

  // Draw viewport indicator (what portion of audio is visible in spectrogram)
  // Only show indicator if spectrogram has been rendered (canvas has content)
  if (canvas.width === 0 || canvas.height === 0) return

  let viewportWidth = canvasContainer.clientWidth || 1920
  let totalCanvasWidth = Math.ceil(frameCount * zoom)

  // Calculate the actual time window being shown
  // If time window >= audio duration, show full indicator
  // Otherwise, calculate based on time window and offset
  let timeWindowSeconds = zoomSeconds
  let showsFullAudio = timeWindowSeconds >= audioDuration

  if (showsFullAudio) {
    // Show indicator covering entire waveform since all frames are rendered
    context.fillStyle = 'rgba(255, 255, 0, 0.2)'
    context.fillRect(0, 0, canvasWidth, canvasHeight)
    context.strokeStyle = '#ff0'
    context.lineWidth = 2
    context.strokeRect(0, 0, canvasWidth, canvasHeight)
  } else {
    // When zoom >= 1 and time window < audio duration, calculate which portion is visible
    // Calculate based on actual time position and window
    let totalDuration = audioDuration
    let timeWindowRatio = timeWindowSeconds / totalDuration

    // Calculate start time from offset
    // offset is in pixels, convert to time
    let totalPixels = Math.ceil(frameCount * zoom)
    let offsetRatio = offset / totalPixels
    let startTimeRatio = offsetRatio

    // Map time ratios to waveform canvas positions
    let indicatorX = startTimeRatio * canvasWidth
    let indicatorWidth = timeWindowRatio * canvasWidth

    // Clamp to waveform canvas bounds
    indicatorX = Math.max(0, Math.min(indicatorX, canvasWidth))
    indicatorWidth = Math.max(
      0,
      Math.min(indicatorWidth, canvasWidth - indicatorX),
    )

    // Draw semi-transparent overlay for visible region
    context.fillStyle = 'rgba(255, 255, 0, 0.2)'
    context.fillRect(indicatorX, 0, indicatorWidth, canvasHeight)

    // Draw border
    context.strokeStyle = '#ff0'
    context.lineWidth = 2
    context.strokeRect(indicatorX, 0, indicatorWidth, canvasHeight)
  }
}

// Convert seconds to internal zoom value
let secondsToZoom = (seconds: number): number => {
  if (!cachedParams) return 1
  let viewportWidth = canvasContainer.clientWidth || 1920
  // zoom = (viewportWidth * hopSize) / (seconds * sampleRate)
  return (
    (viewportWidth * cachedParams.hopSize) / (seconds * cachedParams.sampleRate)
  )
}

// Convert internal zoom value to seconds
let zoomToSeconds = (zoomValue: number): number => {
  if (!cachedParams) return 1
  let viewportWidth = canvasContainer.clientWidth || 1920
  // seconds = (viewportWidth * hopSize) / (zoom * sampleRate)
  return (
    (viewportWidth * cachedParams.hopSize) /
    (zoomValue * cachedParams.sampleRate)
  )
}

let updateZoomDisplay = () => {
  // Update display with seconds, rounded to 3 decimal places for subsecond precision
  let displaySeconds = Math.round(zoomSeconds * 1000) / 1000
  zoomSlider.value = String(displaySeconds)
  zoomInput.value = String(displaySeconds)
}

let setZoomSeconds = (newSeconds: number) => {
  if (!cachedAudioData || !cachedParams) return

  // Clamp seconds to valid range (0.01 to audio duration, no max limit)
  let maxSeconds = audioDuration || Infinity
  newSeconds = Math.max(0.01, Math.min(maxSeconds, newSeconds))

  // If we have a valid viewport, maintain center position when zooming
  if (frameCount > 0) {
    let viewportWidth = canvasContainer.clientWidth || 1920
    let totalPixels = Math.ceil(frameCount * zoom)

    if (totalPixels > 0) {
      // Calculate current center time position before zooming
      let currentCenterPixel = offset + viewportWidth / 2
      let currentCenterRatio = Math.max(
        0,
        Math.min(1, currentCenterPixel / totalPixels),
      )
      let currentCenterTime = currentCenterRatio * audioDuration

      // Update zoom
      zoomSeconds = newSeconds
      zoom = secondsToZoom(zoomSeconds)

      // Calculate new total pixels after zoom change
      let newTotalPixels = Math.ceil(frameCount * zoom)

      // Calculate new center pixel position to maintain the same center time
      let newCenterRatio = currentCenterTime / audioDuration
      let newCenterPixel = newCenterRatio * newTotalPixels

      // Update offset to keep center at the same time position
      offset = Math.max(
        0,
        Math.min(
          newCenterPixel - viewportWidth / 2,
          Math.max(0, newTotalPixels - viewportWidth),
        ),
      )
      updateScrollPosition()
    } else {
      // No valid pixels yet, just update zoom
      zoomSeconds = newSeconds
      zoom = secondsToZoom(zoomSeconds)
    }
  } else {
    // No frame count yet, just update zoom
    zoomSeconds = newSeconds
    zoom = secondsToZoom(zoomSeconds)
  }

  updateZoomDisplay()

  // Don't re-render automatically - user must click "Render Spectrogram" button
  // Just update the viewport indicator
  updateWaveformViewport()
  setStatus('Zoom updated - Click "Render Spectrogram" to apply')
}

let handleZoom = (multiplier: number) => {
  if (!cachedAudioData || !cachedParams || frameCount === 0) return

  // Calculate current center time position before zooming
  let viewportWidth = canvasContainer.clientWidth || 1920
  let totalPixels = Math.ceil(frameCount * zoom)

  // Avoid division by zero
  if (totalPixels === 0) {
    setZoomSeconds(zoomSeconds * multiplier)
    return
  }

  let currentCenterPixel = offset + viewportWidth / 2
  let currentCenterRatio = Math.max(
    0,
    Math.min(1, currentCenterPixel / totalPixels),
  )
  let currentCenterTime = currentCenterRatio * audioDuration

  // Apply zoom by multiplying the time window
  let newZoomSeconds = zoomSeconds * multiplier
  setZoomSeconds(newZoomSeconds)

  // Calculate new zoom and total pixels after zoom change
  let newZoom = secondsToZoom(newZoomSeconds)
  let newTotalPixels = Math.ceil(frameCount * newZoom)

  // Calculate new center pixel position to maintain the same center time
  let newCenterRatio = currentCenterTime / audioDuration
  let newCenterPixel = newCenterRatio * newTotalPixels

  // Update offset to keep center at the same time position
  offset = Math.max(
    0,
    Math.min(
      newCenterPixel - viewportWidth / 2,
      Math.max(0, newTotalPixels - viewportWidth),
    ),
  )
  updateScrollPosition()
  updateWaveformViewport()
}

let handlePan = (delta: number) => {
  let viewportWidth = canvasContainer.clientWidth || 1920
  let canvasWidth = Math.ceil(frameCount * zoom)
  let maxOffset = Math.max(0, canvasWidth - viewportWidth)
  offset = Math.max(0, Math.min(offset + delta, maxOffset))
  updateScrollPosition()
  updateWaveformViewport()
}

let navigateToWaveformPosition = (x: number) => {
  if (!cachedWaveformData || !cachedParams || frameCount === 0) return

  let waveformWidth = waveformCanvas.width
  let positionRatio = x / waveformWidth

  // Calculate target frame based on position
  let targetFrame = Math.floor(positionRatio * frameCount)
  targetFrame = Math.max(0, Math.min(targetFrame, frameCount - 1))

  // Calculate offset to center this frame in viewport
  let viewportWidth = canvasContainer.clientWidth || 1920
  let totalCanvasWidth = Math.ceil(frameCount * zoom)
  let framePosition = (targetFrame / frameCount) * totalCanvasWidth
  offset = Math.max(
    0,
    Math.min(
      framePosition - viewportWidth / 2,
      totalCanvasWidth - viewportWidth,
    ),
  )

  updateScrollPosition()
  updateWaveformViewport()

  // Don't re-render spectrogram automatically - user must click "Render Spectrogram" button
  setStatus(
    'Position updated - Click "Render Spectrogram" to render this region',
  )
}

let isRendering = false
renderSpectrogramBtn.onclick = async () => {
  if (!cachedAudioData || !cachedParams) {
    setStatus('Please load an audio file first')
    return
  }

  // Always abort any previous render when user explicitly clicks render button
  // This allows canceling and restarting when viewport changes
  if (isRendering) {
    abortController.abort()
    // Wait a bit for the abort to propagate
    await new Promise(resolve => setTimeout(resolve, 50))
  }

  // Create new abort controller for this render
  abortController.abort()
  abortController = new AbortController()
  pauseState.paused = false
  updatePauseButtonVisibility()

  isRendering = true
  setStatus(describeProgress({ percent: 0, etaMs: null }))

  try {
    await renderSpectrogram(
      abortController.signal,
      cachedAudioData,
      cachedParams.windowSize,
      cachedParams.hopSize,
      cachedParams.maxFrequency,
    )
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      setStatus('Rendering cancelled - starting new render')
    } else {
      console.error(error)
      setStatus('Failed to render spectrogram')
    }
  } finally {
    isRendering = false
  }
}

pauseBtn.onclick = () => {
  pauseState.paused = true
  updatePauseButtonVisibility()
  setStatus('Processing paused')
}

resumeBtn.onclick = () => {
  pauseState.paused = false
  updatePauseButtonVisibility()
  setStatus('Processing resumed')
}

// Zoom controls
zoomSlider.oninput = () => {
  let value = Number(zoomSlider.value)
  if (Number.isFinite(value)) {
    setZoomSeconds(value)
  }
}

zoomInput.onchange = () => {
  let value = Number(zoomInput.value)
  let maxSeconds = Math.min(audioDuration || 60, 60)
  if (Number.isFinite(value) && value >= 0.01 && value <= maxSeconds) {
    setZoomSeconds(value)
  } else {
    // Reset to current zoom if invalid
    updateZoomDisplay()
  }
}

// Update slider max when audio is loaded
let updateZoomSliderRange = () => {
  if (audioDuration > 0) {
    let maxSeconds = audioDuration
    zoomSlider.max = String(maxSeconds)
    zoomInput.max = String(maxSeconds)
  }
}

zoomInBtn.onclick = () => {
  // Zoom in: decrease time window by 10% (multiply by 0.9)
  handleZoom(0.9)
}
zoomOutBtn.onclick = () => {
  // Zoom out: increase time window by 10% (multiply by 1.1)
  handleZoom(1.1)
}
panLeftBtn.onclick = () => handlePan(-100)
panRightBtn.onclick = () => handlePan(100)

let updateMaxHeight = () => {
  let value = maxHeightInput.value.trim()
  if (value) {
    let height = Number(value)
    if (Number.isFinite(height) && height >= 100) {
      maxHeight = height
      // Don't auto-render - user needs to click "Render Spectrogram" button
      setStatus('Height updated - Click "Render Spectrogram" to apply')
    }
  }
}

input.onchange = run
profileSelect.onchange = () => {
  applyProfileDefaults()
  // Only reload audio and waveform, don't render spectrogram
  run()
}
maxFrequencyInput.onchange = () => {
  // Just update the cached params, don't re-render spectrogram
  // User needs to click "Render Spectrogram" button
  if (cachedParams && cachedAudioData) {
    let mode = profileSelect.value as AudioProfileMode
    let profile = getAudioProfile(mode)
    let maxFrequencyHz = (() => {
      let raw = maxFrequencyInput.value.trim()
      if (!raw) {
        return DEFAULT_MAX_FREQUENCY_HZ
      }
      let parsed = Number(raw)
      if (!Number.isFinite(parsed) || parsed <= 0) {
        return DEFAULT_MAX_FREQUENCY_HZ
      }
      return parsed
    })()
    let binWidth = profile.sampleRate / profile.windowSize
    let maxFrequency = Math.floor(maxFrequencyHz / binWidth)
    let maxAllowedBins = Math.floor(profile.windowSize / 2)
    if (maxFrequency > maxAllowedBins) {
      maxFrequency = maxAllowedBins
    }
    if (maxFrequency < 1) {
      maxFrequency = 1
    }
    cachedParams.maxFrequency = maxFrequency
    setStatus('Settings updated - Click "Render Spectrogram" to apply')
  }
}
maxHeightInput.onchange = updateMaxHeight
maxHeightInput.oninput = updateMaxHeight

// Initialize max height
updateMaxHeight()

// Cursor tracking for timestamp and frequency display
canvas.onmousemove = (event: MouseEvent) => {
  if (!cachedParams) {
    cursorInfoNode.textContent = ''
    return
  }

  let rect = canvas.getBoundingClientRect()
  // Convert CSS coordinates to canvas coordinates (accounting for CSS scaling)
  let scaleX = canvas.width / rect.width
  let scaleY = canvas.height / rect.height
  let x = (event.clientX - rect.left) * scaleX
  let y = (event.clientY - rect.top) * scaleY

  // Clamp to canvas bounds
  x = Math.max(0, Math.min(x, canvas.width))
  y = Math.max(0, Math.min(y, canvas.height))

  // Calculate frame index from x position
  // The canvas shows frames from renderedFrameStart to renderedFrameEnd
  // Map x position to a frame within that range
  let canvasWidth = canvas.width
  let renderedFrameCount = renderedFrameEnd - renderedFrameStart
  let frameRatio = x / canvasWidth
  let frameIndexInRenderedRange = Math.floor(frameRatio * renderedFrameCount)

  // Convert to absolute frame index in the full audio
  let absoluteFrame = renderedFrameStart + frameIndexInRenderedRange
  absoluteFrame = Math.max(
    renderedFrameStart,
    Math.min(absoluteFrame, renderedFrameEnd - 1),
  )

  // Calculate absolute timestamp in the source audio
  let sampleIndex = absoluteFrame * cachedParams.hopSize
  let timestamp = sampleIndex / cachedParams.sampleRate

  // Calculate frequency from y position
  // Canvas y=0 is at top (highest frequency), y=height is at bottom (lowest frequency)
  // Account for vertical scaling if canvas height is less than maxFrequency
  let verticalScale =
    canvas.height < cachedParams.maxFrequency
      ? canvas.height / cachedParams.maxFrequency
      : 1

  // Map canvas y to frequency bin index
  let freqIndexStart = (canvas.height - 1 - y) / verticalScale
  let freqIndexEnd = (canvas.height - y) / verticalScale
  let freqStart = Math.floor(freqIndexStart)
  let freqEnd = Math.ceil(freqIndexEnd)

  // Clamp to valid range
  freqStart = Math.max(0, Math.min(freqStart, cachedParams.maxFrequency - 1))
  freqEnd = Math.max(0, Math.min(freqEnd, cachedParams.maxFrequency))

  // Use the center frequency bin for display
  let frequencyBinIndex = Math.floor((freqStart + freqEnd) / 2)

  // Calculate frequency range for this bin
  // Each bin represents a frequency range
  // The FFT produces windowSize/2 bins, each representing sampleRate/windowSize Hz
  // binWidth = sampleRate / windowSize (Hz per bin)
  let binWidth = cachedParams.sampleRate / cachedParams.windowSize

  // Calculate the actual frequency for this bin
  // maxFrequency is the number of bins displayed (0 to maxFrequency-1)
  let frequencyStart = frequencyBinIndex * binWidth
  let frequencyEnd = (frequencyBinIndex + 1) * binWidth

  // Format display - show both mm:ss.sss and seconds format
  let timeStr = formatTimeBoth(timestamp)
  let freqStr = `${frequencyStart.toFixed(1)} - ${frequencyEnd.toFixed(1)} Hz`

  cursorInfoNode.textContent = `Time: ${timeStr} | Frequency: ${freqStr}`

  // Update hover timestamp and draw indicator on overlay canvas
  hoverTimestamp = timestamp
  updateWaveformHoverIndicator()
}

canvas.onmouseleave = () => {
  cursorInfoNode.textContent = ''
  hoverTimestamp = null
  updateWaveformHoverIndicator()
}

// Update hover indicator on overlay canvas (separate from waveform redraw)
let updateWaveformHoverIndicator = () => {
  if (!cachedWaveformData || !cachedParams || frameCount === 0) return

  let overlayContext = waveformOverlayCanvas.getContext('2d')!
  let dpr = window.devicePixelRatio || 1
  // Reapply context scale to ensure it's set (context scale persists, but be explicit)
  overlayContext.setTransform(dpr, 0, 0, dpr, 0, 0)

  // Use logical dimensions (same as updateWaveformViewport uses for calculations)
  // Since context is scaled by dpr, we use logical dimensions
  let canvasWidth = waveformCanvas.width / dpr
  let canvasHeight = waveformCanvas.height / dpr

  // Clear overlay canvas (use logical dimensions since context is scaled)
  overlayContext.clearRect(0, 0, canvasWidth, canvasHeight)

  if (hoverTimestamp === null || audioDuration <= 0) {
    // Already cleared, just return
    return
  }

  // Only show indicator if spectrogram has been rendered
  if (canvas.width === 0 || canvas.height === 0) return

  // Calculate the yellow box (viewport) position and size
  let viewportWidth = canvasContainer.clientWidth || 1920
  let totalCanvasWidth = Math.ceil(frameCount * zoom)
  let timeWindowSeconds = zoomSeconds
  let showsFullAudio = timeWindowSeconds >= audioDuration

  let indicatorX: number
  let indicatorWidth: number

  if (showsFullAudio) {
    // Full audio is shown - yellow box covers entire waveform
    indicatorX = 0
    indicatorWidth = canvasWidth
  } else {
    // Calculate yellow box position
    let totalDuration = audioDuration
    let timeWindowRatio = timeWindowSeconds / totalDuration
    let totalPixels = Math.ceil(frameCount * zoom)
    let offsetRatio = offset / totalPixels
    let startTimeRatio = offsetRatio

    indicatorX = startTimeRatio * canvasWidth
    indicatorWidth = timeWindowRatio * canvasWidth

    // Clamp to waveform canvas bounds
    indicatorX = Math.max(0, Math.min(indicatorX, canvasWidth))
    indicatorWidth = Math.max(
      0,
      Math.min(indicatorWidth, canvasWidth - indicatorX),
    )
  }

  // Calculate the time range of the yellow box
  let viewportStartTime = (indicatorX / canvasWidth) * audioDuration
  let viewportEndTime =
    ((indicatorX + indicatorWidth) / canvasWidth) * audioDuration
  let viewportTimeRange = viewportEndTime - viewportStartTime

  // Check if hover timestamp is within the viewport
  if (
    hoverTimestamp >= viewportStartTime &&
    hoverTimestamp <= viewportEndTime
  ) {
    // Calculate relative position within the viewport
    let relativeTime = hoverTimestamp - viewportStartTime
    let relativeRatio =
      viewportTimeRange > 0 ? relativeTime / viewportTimeRange : 0

    // Map to position within the yellow box
    let hoverX = indicatorX + relativeRatio * indicatorWidth
    hoverX = Math.max(indicatorX, Math.min(hoverX, indicatorX + indicatorWidth))

    // Draw vertical line (thin line, 1px)
    overlayContext.strokeStyle = '#0ff'
    overlayContext.lineWidth = 1
    overlayContext.beginPath()
    overlayContext.moveTo(hoverX, 0)
    overlayContext.lineTo(hoverX, canvasHeight)
    overlayContext.stroke()
  }
}

// Waveform navigation
let isDraggingWaveform = false
waveformCanvas.onmousedown = (event: MouseEvent) => {
  if (!cachedWaveformData) return
  isDraggingWaveform = true
  let rect = waveformCanvas.getBoundingClientRect()
  let x = event.clientX - rect.left
  navigateToWaveformPosition(x)
}

waveformCanvas.onmousemove = (event: MouseEvent) => {
  if (!cachedWaveformData || !cachedParams) {
    cursorInfoNode.textContent = ''
    return
  }

  let rect = waveformCanvas.getBoundingClientRect()
  let scaleX = waveformCanvas.width / rect.width
  let x = (event.clientX - rect.left) * scaleX

  // Clamp to canvas bounds
  x = Math.max(0, Math.min(x, waveformCanvas.width))

  // If dragging, navigate to position
  if (isDraggingWaveform) {
    navigateToWaveformPosition(x)
    return
  }

  // Otherwise, show timestamp info
  // Calculate timestamp from x position
  // x position maps to a position in the waveform data
  let positionRatio = x / waveformCanvas.width
  let sampleIndex = Math.floor(positionRatio * cachedAudioData!.length)
  sampleIndex = Math.max(0, Math.min(sampleIndex, cachedAudioData!.length - 1))
  let timestamp = sampleIndex / cachedParams.sampleRate

  // Format display - show both mm:ss.sss and seconds format
  let timeStr = formatTimeBoth(timestamp)
  cursorInfoNode.textContent = `Time: ${timeStr}`
}

waveformCanvas.onmouseup = () => {
  isDraggingWaveform = false
}

waveformCanvas.onmouseleave = () => {
  isDraggingWaveform = false
  cursorInfoNode.textContent = ''
}

run()

function querySelector<E extends HTMLElement>(selector: string) {
  let element = document.querySelector<E>(selector)
  if (!element) {
    throw new Error(`Element not found: "${selector}"`)
  }
  return element
}
