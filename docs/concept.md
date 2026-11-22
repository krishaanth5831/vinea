# Vinea — Concept (as of June 2026)

> This is the concept the repo was started on. The company's framing has moved
> on since (see [getvinea.nl](https://getvinea.nl)), and the robot work now
> lives in separate per-module repos. Kept for context on why the simulation
> was built the way it was.

## The Problem

Harvesting and scouting in Dutch commercial greenhouses are done by hand, and
the seasonal labour that does them is getting harder to find every year.
A robot that helps has to fit the greenhouse as it already is, because
rebuilding a glasshouse around a machine is not an option for most growers.

## The Solution

A modular autonomous robot that works inside **existing** greenhouse row
infrastructure, with no rebuild required.

Two swappable modules on one configurable base:
- **Harvest module** — autonomous picking and de-leafing, camera-guided,
  crop-specific grippers (tomato first)
- **Scout module** — routine plant-health inspection, aiming to catch disease
  and pests before they are visible to a person walking the row

Same chassis across all customers. Only the gripper, the row-width adapter and
the module set vary.

## Why simulation first

The single most important technical question was whether one arm on a moving
base can pick fast enough, and cleanly enough, to be worth deploying. That can
be answered in a physics simulator before any hardware is bought, which is what
`simulation/mujoco/` does. The results, including the ones that came out badly,
are in [results.md](results.md).
