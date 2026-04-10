/**
 * Lightweight logger that silences debug/info output in production builds.
 * Errors and warnings always print regardless of build mode.
 */

const isDev = import.meta.env.DEV

export const logger = {
  debug: isDev ? console.debug.bind(console) : () => {},
  info: isDev ? console.info.bind(console) : () => {},
  warn: console.warn.bind(console),
  error: console.error.bind(console),
}
