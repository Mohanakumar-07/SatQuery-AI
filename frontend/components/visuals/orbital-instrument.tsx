import { Radio, Satellite } from 'lucide-react';

export function OrbitalInstrument() {
  return (
    <div className="orbital-instrument" aria-label="Satellite scanning Earth">
      <svg aria-hidden="true" className="orbit-geometry" viewBox="0 0 640 640" fill="none">
        <circle cx="320" cy="320" r="232" className="orbit-ring orbit-ring-main" />
        <circle cx="320" cy="320" r="170" className="orbit-ring orbit-ring-inner" />
        <path d="M87 340C180 158 436 94 572 268" className="orbit-arc" />
        <path d="M132 468C262 566 468 520 548 366" className="orbit-arc orbit-arc-muted" />
        <path d="M320 150V490M150 320H490" className="instrument-axis" />
        <circle cx="320" cy="320" r="5" className="instrument-node" />
        <circle cx="470" cy="147" r="4" className="instrument-node instrument-node-gold" />
      </svg>
      <div className="earth-contour">
        <svg aria-hidden="true" viewBox="0 0 240 240" fill="none">
          <circle cx="120" cy="120" r="92" />
          <ellipse cx="120" cy="120" rx="44" ry="92" />
          <path d="M28 120H212M48 72H192M48 168H192" />
          <path d="M77 42C91 61 95 75 88 91C80 111 89 124 109 132C131 142 139 161 129 190" />
        </svg>
      </div>
      <div className="satellite-node"><Satellite aria-hidden="true" strokeWidth={1.15} /></div>
      <div className="instrument-label instrument-label-top">
        <span>ORB / 547 KM</span><span>PASS 06:42</span>
      </div>
      <div className="instrument-label instrument-label-bottom">
        <Radio aria-hidden="true" /><span>DOWNLINK STABLE</span>
      </div>
    </div>
  );
}
