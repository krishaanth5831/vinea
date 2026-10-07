# SO-101 test greenhouse — fully printed

A tabletop greenhouse for the SO-101 hardware track: three rows of hanging cherry
tomato vines, aisles the arm stands in, and nothing overhead or at the ends that the
arm can hit at full stretch. Every piece is printed; a 256 × 256 mm bed fits all of them.

Onshape (private, parametric): **SO-101 Greenhouse** —
<https://cad.onshape.com/documents/54620f37090a501ebf4a1a7f>. Double-click the
*SO-101 Greenhouse* feature to change any dimension or fit; the source is
`so101_greenhouse.fs` here.

## Size and layout

| | |
|---|---|
| Footprint | 760 × 820 mm (along rows × across) |
| Height | 692 mm to the ridge |
| Rows | 3, at 400 mm pitch (y = −400 / 0 / +400) |
| Aisles | 2, at y = ±200, so every vine the arm works is 200 mm away |
| Crop rails (vine tops) | 520 mm, above the arm's ~490 mm vertical reach |
| Arm position | middle of an aisle, 370 mm from each end frame |

```
      end frame            end frame
   y=+400  ===rail==============   row 1   (vines hang here)
   y=+200   . . . [SO-101] . . .   aisle A  - arm picks rows 1 and 2
   y=   0  ===rail==============   row 2
   y=-200   . . . . . . . . . .    aisle B  - move the arm here for rows 2 and 3
   y=-400  ===rail==============   row 3
```

**Free articulation was checked in the model**, not assumed. A sphere of the arm's
full reach (400 mm from the shoulder incl. the jaw, shoulder 117 mm up, from the
SO-ARM100 URDF) sits in the aisle. ⚠️ Only table-level pieces (feet, anchor rails,
base ties) fall inside it — they're the floor. Every post, tie, rafter and crop rail
stays at least 7 mm outside. At 720 mm long the four nearest posts grazed the
sphere by 1–2 mm, which is why the house is 740 mm between end frames.

## Parts (print these)

| STL | Qty | Size (mm) | Notes |
|---|---|---|---|
| `gh_fit_coupon.stl` | 1 | 76 × 46 × 26 | **Print first** — see Fit below |
| `gh_pin_x86.stl` | 86 | 40 × 16.5 × 16.5 | Every joint; print 20–30 per plate |
| `gh_rail_240.0_x18.stl` | 18 | 240 × 20 × 20 | Crop + anchor rails, keyholes every 20 mm |
| `gh_tube_240.0_x3.stl` | 3 | 240 × 20 × 20 | Ridge |
| `gh_tube_220.0_x12.stl` | 12 | 220 × 20 × 20 | Posts (2 per post) |
| `gh_tube_156.5_x24.stl` | 24 | 156.5 × 20 × 20 | Base ties, crop ties, rafters (2 each) |
| `gh_tube_101.6_x2.stl` | 2 | 101.6 × 20 × 20 | King post, top |
| `gh_node_F_out_x4.stl` | 4 | 80 × 40 × 20 | Outer feet, with a table tab (4.5 mm screw/clamp hole) |
| `gh_node_F_mid_x2.stl` | 2 | 87 × 40 × 20 | Middle feet |
| `gh_node_C_out_x4.stl` | 4 | 91 × 80 × 20 | Eave corners: post, tie, rafter, crop rail |
| `gh_node_C_mid_x2.stl` | 2 | 70 × 60 × 20 | King post crossing at crop height |
| `gh_node_R_x2.stl` | 2 | 72 × 40 × 20 | Ridge |

160 pieces, about **2.5 kg PLA** as printed (3.3 kg if everything were solid), roughly
50–60 h on a Bambu-class printer.

**Print settings:** PLA, 0.4 mm nozzle, 0.2 mm layers, **4 walls** (the tubes are
1.6 mm thick, so this prints them solid), 15 % infill. All parts are already in
print orientation and need **no supports**. Nodes lie flat in their own plane;
the square holes print as short bridges. Every edge is rounded.

## Fit (print the coupon first)

- **Pin clearance**: three sockets, 1 / 2 / 3 dots = bore −0.1 / nominal / +0.1 mm.
  Push a printed pin into each. The right one goes in by hand and doesn't wobble.
  If that's not the 2-dot socket, change *Pin clearance* in Onshape and re-export.
  Default is 0.15 mm per side (pin 16.5 mm in a 16.8 mm bore).
- **Keyhole slot**: three slots at 1.5 / 1.8 / 2.2 mm. Your twine must slide into
  the slot and a knot must **not** pull through it. Set *Keyhole slot width* to match.

## Assembly

1. **End frames (×2, identical).** Feet at the bottom, outer posts up to the eave
   corners, king post up the middle through the crop-height node to the ridge.
   Then the base ties, crop ties and rafters. Every joint is a pin pushed halfway
   into each side. Tubes join end to end with a pin too.
2. **Long rails.** Join the two end frames with 3 anchor rails on the table, 3 crop
   rails at 520 mm and the ridge. Each runs straight through the node's square
   through-hole socket.
3. **Lock bolts (optional).** Every socket and tube end has a 3.4 mm hole at the
   middle of the pin. Use M3 × 25 + nut where a joint feels loose, or glue joints
   you never want to take apart.
4. **Fix it to the table.** Screw or clamp the 4 outer feet through their tabs.
   ⚠️ The vault rig rule: the frame must not move when the gripper closes. A clamped
   frame doesn't.

The rails run in different directions on the two levels. Anchor rails go keyholes
**up**, crop rails keyholes **down**. The tick marks on the rails' sides line up
top and bottom.

## Hanging vines

- Use horticultural twine with a knot (or bead) at each end.
- **Top:** push the knot up through a crop-rail keyhole's round entry and slide it
  into the slot. **Bottom:** do the same in the anchor-rail keyhole directly below,
  with the twine pulled taut. A taut vine doesn't swing away when the jaw closes,
  which single-fruit picking needs.
- Keyholes are every 20 mm. Keep neighbouring vines at least 60 mm apart so the open
  jaw never brushes the next one; the model shows vines at 300 / 360 / 420 mm.
- Clip trusses (tomato truss clips) between **100 and 350 mm** above the table, the
  arm's comfortable band from the aisle.
- Re-hanging a vine at a named keyhole takes seconds, which meets the vault rule of
  re-hanging under 60 s.

## Arm and cameras

- Fix the SO-101 base mid-aisle, 370 mm from either end, centred on y = ±200. Mark
  it on the table, and mark where the feet sit.
- ⚠️ Per the vault recording notes, the front (birdseye) camera clamps to the
  **table**, not to this frame — the frame gets bumped.
- The structure works for cutting trusses (the vault plan) as well as plucking
  single fruit.
