'use client';

type SceneMapProps = {
  showMask?: boolean;
  showRegions?: boolean;
  compact?: boolean;
  baseImageUrl?: string | null;
  overlayUrl?: string | null;
  overlayAlt?: string;
  coordinateLabel?: string | null;
};

export function SceneMap({
  showMask = true,
  showRegions = true,
  compact = false,
  baseImageUrl,
  overlayUrl,
  overlayAlt = 'Analysis evidence overlay',
  coordinateLabel,
}: SceneMapProps) {
  const hasImage = Boolean(baseImageUrl || overlayUrl);

  return (
    <div className={compact ? 'scene-map scene-map-compact' : 'scene-map'}>
      {baseImageUrl && (
        <img
          className="scene-map-base"
          src={baseImageUrl}
          alt="Input satellite scene"
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            backgroundColor: '#0a0d10',
            zIndex: 1,
          }}
        />
      )}
      {overlayUrl && (
        <img
          className="scene-map-evidence"
          src={overlayUrl}
          alt={overlayAlt}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            zIndex: 2,
            mixBlendMode: 'screen',
          }}
        />
      )}
      <svg
        viewBox="0 0 800 520"
        role="img"
        aria-label="Satellite analysis spatial viewer"
        style={{ position: 'relative', zIndex: 3, pointerEvents: 'none' }}
      >
        {!hasImage && <rect width="800" height="520" className="map-base" />}
        <g className="map-grid" opacity={hasImage ? 0.35 : 1}>
          {Array.from({ length: 9 }).map((_, index) => (
            <path key={`v${index}`} d={`M${index * 100} 0V520`} />
          ))}
          {Array.from({ length: 7 }).map((_, index) => (
            <path key={`h${index}`} d={`M0 ${index * 86.7}H800`} />
          ))}
        </g>
        {!hasImage && (
          <>
            <path className="terrain-line" d="M-20 404C95 333 145 364 223 312C292 266 338 276 402 218C480 146 552 181 620 111C685 44 742 73 823 26" />
            <path className="terrain-line terrain-line-2" d="M-34 478C98 408 176 447 282 374C362 319 421 345 503 279C607 196 669 226 827 143" />
            <path className="water-line" d="M32 67C108 112 94 177 157 206C227 238 204 304 281 330C355 355 367 430 447 466C505 492 574 475 629 525" />
            <g className="parcel-lines">
              <path d="M97 24L252 170L162 300L21 202Z" />
              <path d="M267 31L413 79L369 213L244 169Z" />
              <path d="M441 72L616 89L583 234L371 215Z" />
              <path d="M620 90L780 39L808 190L584 234Z" />
              <path d="M163 302L282 332L252 481L49 454Z" />
              <path d="M286 331L449 468L340 527L250 480Z" />
            </g>
          </>
        )}
        {showMask && !hasImage && (
          <g className="change-mask">
            <path d="M465 156L565 130L628 187L601 274L506 286L438 229Z" />
            <path d="M276 344L367 329L424 379L397 449L309 466L253 405Z" />
          </g>
        )}
        {showRegions && !hasImage && (
          <g className="region-outlines">
            <path d="M465 156L565 130L628 187L601 274L506 286L438 229Z" />
            <path d="M276 344L367 329L424 379L397 449L309 466L253 405Z" />
            <circle cx="531" cy="211" r="8" />
            <circle cx="338" cy="397" r="8" />
          </g>
        )}
        <g className="map-reticle" opacity={hasImage ? 0.6 : 1}>
          <path d="M400 224V296M364 260H436" />
          <circle cx="400" cy="260" r="21" />
        </g>
      </svg>
      <div className="map-coordinate map-coordinate-top">
        {coordinateLabel ?? (hasImage ? 'INPUT SCENE / VISUAL EVIDENCE' : 'ILLUSTRATIVE SCENE')}
      </div>
      <div className="map-coordinate map-coordinate-bottom">
        {overlayUrl ? 'OVERLAY MASK ACTIVE' : baseImageUrl ? 'BASE SATELLITE IMAGERY' : 'NO BACKEND ARTIFACT'}
      </div>
      <div className="map-north">N<span>↑</span></div>
      <div className="scan-line" />
    </div>
  );
}
