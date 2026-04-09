// Shared axios client (default export preserves `import api from '../api'`)
export { api } from './client'
export { api as default } from './client'

// Domain modules
export * from './cameras'
export * from './alerts'
export * from './baselines'
export * from './models'
export * from './training'
export * from './system'
export * from './users'
export * from './replay'
export * from './tasks'
