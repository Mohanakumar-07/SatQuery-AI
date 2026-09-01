'use client';

import { useEffect, useMemo, useState } from 'react';
import { geoGraticule10, geoOrthographic, geoPath } from 'd3-geo';
import { feature, mesh } from 'topojson-client';
import type { GeometryCollection, Topology } from 'topojson-specification';
import worldData from 'world-atlas/countries-110m.json';

type WorldObjects = {
  countries: GeometryCollection;
  land: GeometryCollection;
};

const topology = worldData as unknown as Topology<WorldObjects>;
const land = feature(topology, topology.objects.land);
const borders = mesh(topology, topology.objects.countries, (a, b) => a !== b);
const graticule = geoGraticule10();

export function RotatingEarth() {
  const [longitude, setLongitude] = useState(-12);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let frameId = 0;
    let lastUpdate = 0;
    const rotate = (time: number) => {
      if (time - lastUpdate > 45) {
        setLongitude((value) => (value + 0.12) % 360);
        lastUpdate = time;
      }
      frameId = requestAnimationFrame(rotate);
    };
    frameId = requestAnimationFrame(rotate);
    return () => cancelAnimationFrame(frameId);
  }, []);

  const paths = useMemo(() => {
    const projection = geoOrthographic()
      .translate([500, 500])
      .scale(454)
      .rotate([longitude, -28, -6])
      .clipAngle(90)
      .precision(0.3);
    const path = geoPath(projection);

    return {
      sphere: path({ type: 'Sphere' }),
      graticule: path(graticule),
      land: path(land),
      borders: path(borders),
    };
  }, [longitude]);

  return (
    <svg
      className="rotating-earth"
      viewBox="0 0 1000 1000"
      role="img"
      aria-label="Slowly rotating wireframe Earth"
    >
      <defs>
        <radialGradient id="earth-shade" cx="34%" cy="28%" r="72%">
          <stop offset="0" stopColor="#20262a" stopOpacity=".42" />
          <stop offset=".55" stopColor="#0c0f12" stopOpacity=".22" />
          <stop offset="1" stopColor="#020304" stopOpacity=".92" />
        </radialGradient>
        <filter id="earth-soft-glow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <path d={paths.sphere ?? undefined} className="earth-disc" />
      <path d={paths.graticule ?? undefined} className="earth-graticule" />
      <path d={paths.land ?? undefined} className="earth-land" />
      <path d={paths.borders ?? undefined} className="earth-borders" />
      <path d={paths.sphere ?? undefined} className="earth-rim" filter="url(#earth-soft-glow)" />
    </svg>
  );
}
