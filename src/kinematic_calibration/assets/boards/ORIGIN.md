These legacy printable and simulator assets are copied without modification from
`/home/ubuntu/workspace/dexforce/auto_calibrate_sim/handeye_calib_sim/assets`
at commit `9488a2dbffc3e7635f3b0109a8839bd261a57cb0`.

The six-marker `A4-300_300_3` profile is a **300 × 300 mm** square board, with
130 mm column pitch and 65 mm physical row pitch. The origin of the detected
board is the first marker center, 52 mm from the top-left edges. Its OBJ origin
is the center of the square; `BoardConfig.board_T_mesh` handles this difference.

The single-marker OBJ is 120 × 120 mm. Its legacy SVG uses pixel/viewBox units,
so print it at an explicitly measured physical scale. A single circular marker
does not establish in-plane orientation and cannot provide a full pose alone.

These retained CCTag templates have legacy ring profiles. The HandEye toolkit
submodule includes its own maintained versioned templates. Match the detector
profile to the physical board; geometric dimensions alone do not establish
ring-profile compatibility. With the toolkit's staggered-grid convention, pass
`board_size=(2, 3)` and `board_spacing=(0.13, 0.13)` for the legacy multi-board:
the second detector parameter is **twice** the 65 mm physical row pitch.

No new licensing terms are asserted for these existing project assets.
