/**
 * Which Yatrikart AI module this dashboard is. Litter Detection and Uniform Compliance are
 * separate applications with separate models; the module only drives the product name shown
 * under the YATRIKART brand (sidebar, browser tab).
 *
 * Set NEXT_PUBLIC_APP_MODULE in .env.local (litter | uniform). Defaults to litter - this
 * repository is the Litter Detection application.
 */
export type AppModule = 'uniform' | 'litter'

export const MODULES: Record<AppModule, { name: string }> = {
  uniform: { name: 'Uniform Compliance' },
  litter: { name: 'Litter Detection' },
}

const env = process.env.NEXT_PUBLIC_APP_MODULE as AppModule | undefined
export const ACTIVE_MODULE: AppModule = env && env in MODULES ? env : 'litter'
export const MODULE_NAME = MODULES[ACTIVE_MODULE].name
export const BRAND = 'Yatrikart'
