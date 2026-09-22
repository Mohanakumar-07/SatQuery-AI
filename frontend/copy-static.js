import fs from 'node:fs';
import path from 'node:path';

const src = path.resolve('dist/server/prerendered-routes');
const dest = path.resolve('dist/client');

if (fs.existsSync(src)) {
  fs.cpSync(src, dest, { recursive: true });
  console.log('[copy-static] Successfully copied prerendered static HTML routes into dist/client');
} else {
  console.warn('[copy-static] Warning: prerendered-routes directory not found at', src);
}
