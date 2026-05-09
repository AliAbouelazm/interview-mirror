import { useEffect, useRef, useState } from 'react'

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

let cachedLandmarker = null
let cachedLoader = null

async function getLandmarker() {
  if (cachedLandmarker) return cachedLandmarker
  if (!cachedLoader) {
    cachedLoader = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
      const lm = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: 'VIDEO',
        numFaces: 1,
      })
      cachedLandmarker = lm
      return lm
    })()
  }
  return cachedLoader
}

function pickBlendshape(blendshapes, name) {
  if (!blendshapes || !blendshapes.length) return 0
  const cat = blendshapes[0]?.categories?.find(c => c.categoryName === name)
  return cat ? cat.score : 0
}

function deriveMetricsFromResult(result) {
  const lms = result.faceLandmarks?.[0]
  const blendshapes = result.faceBlendshapes
  const matrix = result.facialTransformationMatrixes?.[0]?.data
  if (!lms) {
    return null
  }

  let yaw = 0, pitch = 0, roll = 0
  if (matrix && matrix.length === 16) {
    const r00 = matrix[0], r01 = matrix[1], r02 = matrix[2]
    const r10 = matrix[4], r12 = matrix[6]
    const r20 = matrix[8], r21 = matrix[9], r22 = matrix[10]
    yaw = Math.atan2(-r20, Math.sqrt(r00 * r00 + r10 * r10))
    pitch = Math.atan2(r21, r22)
    roll = Math.atan2(r10, r00)
  }

  const eyeBlinkLeft = pickBlendshape(blendshapes, 'eyeBlinkLeft')
  const eyeBlinkRight = pickBlendshape(blendshapes, 'eyeBlinkRight')
  const eyeOpenness = 1 - Math.max(eyeBlinkLeft, eyeBlinkRight)

  const smileLeft = pickBlendshape(blendshapes, 'mouthSmileLeft')
  const smileRight = pickBlendshape(blendshapes, 'mouthSmileRight')
  const smile = (smileLeft + smileRight) / 2

  const lookingAtCamera = Math.max(0, 1 - (Math.abs(yaw) + Math.abs(pitch)) / 0.6)

  return {
    landmarks: lms,
    head_yaw: +yaw.toFixed(4),
    head_pitch: +pitch.toFixed(4),
    head_roll: +roll.toFixed(4),
    eye_openness: +eyeOpenness.toFixed(3),
    smile: +smile.toFixed(3),
    looking_at_camera: +lookingAtCamera.toFixed(3),
  }
}

export function useFaceMesh({ video, enabled = true, onTick }) {
  const [ready, setReady] = useState(false)
  const [latest, setLatest] = useState(null)
  const rafRef = useRef(null)
  const lastTimestampRef = useRef(0)

  useEffect(() => {
    if (!enabled || !video) return undefined
    let cancelled = false

    getLandmarker()
      .then((lm) => {
        if (cancelled) return
        setReady(true)

        const tick = () => {
          if (cancelled) return
          if (video.readyState >= 2) {
            const ts = performance.now()
            if (ts > lastTimestampRef.current) {
              lastTimestampRef.current = ts
              try {
                const result = lm.detectForVideo(video, ts)
                const metrics = deriveMetricsFromResult(result)
                if (metrics) {
                  setLatest(metrics)
                  onTick && onTick(metrics)
                }
              } catch {}
            }
          }
          rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
      })
      .catch(() => setReady(false))

    return () => {
      cancelled = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [video, enabled, onTick])

  return { ready, latest }
}
