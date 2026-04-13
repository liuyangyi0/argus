const fs = require('fs')
const path = require('path')

const dir = 'c:/Users/here/code/argus/web/src'

const map = {
  'var(--argus-text)': 'var(--ink-2)',
  'var(--argus-text-muted)': 'var(--ink-4)',
  'var(--argus-border)': 'var(--line-2)',
  'var(--argus-sidebar-border)': 'var(--line-2)',
  'var(--argus-card-bg)': 'var(--glass)',
  'var(--argus-card-bg-solid)': 'var(--bg)',
  'var(--argus-surface)': 'var(--glass)',
  'var(--argus-header-bg)': 'transparent',
  'var(--argus-footer-bg)': 'transparent',
  'var(--argus-hover-bg)': 'rgba(10, 10, 15, 0.05)',
  'var(--argus-icon-muted)': 'var(--ink-5)',
  'var(--argus-font-mono)': "'JetBrains Mono', ui-monospace, monospace"
}

function processDir(dirPath) {
  const files = fs.readdirSync(dirPath)
  for (const file of files) {
    const fullPath = path.join(dirPath, file)
    if (fs.statSync(fullPath).isDirectory()) {
      processDir(fullPath)
    } else if (fullPath.endsWith('.vue') || fullPath.endsWith('.css') || fullPath.endsWith('.ts')) {
      let content = fs.readFileSync(fullPath, 'utf8')
      let changed = false
      for (const [oldVar, newVar] of Object.entries(map)) {
        if (content.includes(oldVar)) {
          content = content.replace(new RegExp(oldVar.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), newVar)
          changed = true
        }
      }
      if (changed) {
        fs.writeFileSync(fullPath, content, 'utf8')
        console.log('Updated:', fullPath)
      }
    }
  }
}

processDir(dir)
console.log('Migration complete.')
