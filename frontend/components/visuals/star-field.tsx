export function StarField({ dense = false }: { dense?: boolean }) {
  const stars = [
    [4, 14, 1], [10, 66, 1.4], [17, 30, 0.8], [23, 82, 1],
    [31, 10, 1.2], [38, 51, 0.8], [46, 22, 1.1], [54, 73, 0.7],
    [62, 12, 1], [69, 43, 1.4], [76, 88, 0.9], [83, 25, 0.7],
    [91, 59, 1.1], [96, 8, 0.8], [88, 78, 1.3], [58, 91, 0.8],
    ...(dense ? [[7, 91, .7], [27, 48, .8], [42, 94, 1], [72, 61, .7], [98, 36, 1]] : []),
  ];

  return (
    <svg aria-hidden="true" className="star-field" viewBox="0 0 100 100" preserveAspectRatio="none">
      {stars.map(([cx, cy, r], index) => (
        <circle
          key={`${cx}-${cy}`}
          cx={cx}
          cy={cy}
          r={r / 9}
          className={`star star-${(index % 3) + 1}`}
        />
      ))}
    </svg>
  );
}
