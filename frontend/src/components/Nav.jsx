import { Link, useLocation } from 'react-router-dom'
import styles from '../styles/components.module.css'

export default function Nav() {
  const loc = useLocation()
  const onSession = loc.pathname.startsWith('/session')
  return (
    <nav className={styles.nav}>
      <Link to="/" className={styles.brand}>Interview Mirror</Link>
      <div className={styles.navRight}>
        {onSession && <span className={styles.navState}>Live session</span>}
      </div>
    </nav>
  )
}
