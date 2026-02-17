export const cockpitFlags = {
  inspector: import.meta.env.VITE_COCKPIT_INSPECTOR !== 'false',
  elkLayout: import.meta.env.VITE_COCKPIT_ELK_LAYOUT !== 'false',
  c4RenderedDiff: import.meta.env.VITE_COCKPIT_C4_RENDERED_DIFF !== 'false',
}
