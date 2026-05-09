import { useEffect, useRef } from 'react'

const TESSELATION_PAIRS = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251],
  [251, 389], [389, 356], [356, 454], [454, 323], [323, 361],
  [361, 288], [288, 397], [397, 365], [365, 379], [379, 378],
  [378, 400], [400, 377], [377, 152], [152, 148], [148, 176],
  [176, 149], [149, 150], [150, 136], [136, 172], [172, 58],
  [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
  [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
  // Eyes
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
  [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159],
  [159, 158], [158, 157], [157, 173], [173, 133],
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381],
  [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386],
  [386, 385], [385, 384], [384, 398], [398, 362],
  // Lips outer
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40],
  [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270],
  [270, 409], [409, 291],
  // Eyebrows
  [70, 63], [63, 105], [105, 66], [66, 107],
  [336, 296], [296, 334], [334, 293], [293, 300],
  // Nose ridge
  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19],
]

export default function FaceMeshOverlay({ landmarks, width, height, color = 'rgba(91,109,255,0.9)', lineColor = 'rgba(91,109,255,0.35)' }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!landmarks || landmarks.length === 0) return

    ctx.lineWidth = 1
    ctx.strokeStyle = lineColor
    ctx.beginPath()
    for (const [a, b] of TESSELATION_PAIRS) {
      const pa = landmarks[a]
      const pb = landmarks[b]
      if (!pa || !pb) continue
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height)
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height)
    }
    ctx.stroke()

    ctx.fillStyle = color
    for (const p of landmarks) {
      ctx.beginPath()
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.1, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [landmarks, color, lineColor])

  return (
    <canvas
      ref={ref}
      width={width || 640}
      height={height || 360}
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    />
  )
}
