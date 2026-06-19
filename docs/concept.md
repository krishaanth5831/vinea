# Vinea — Concept

## The Problem

Dutch greenhouse operators spend ~€1.6M/year per 5-hectare site on manual harvesting and scouting labour. Seasonal labour supply is structurally collapsing — 75% less available greenhouse labour reported recently. The only funded competitor (SAIA Agrobotics) requires a full greenhouse rebuild, leaving ~95% of the market with no viable automation option.

## The Solution

A modular autonomous robot that works inside **existing** greenhouse row infrastructure — no rebuild required.

Two swappable modules on one configurable base:
- **Harvest module** — autonomous picking + de-leafing, CV-guided, crop-specific grippers (tomato first)
- **Scout module** — daily plant health inspection, disease/pest detection 3–5 days before visible symptoms, grower dashboard alerts

Same chassis across all customers. Only the gripper, row-width adapter, and licensed module set vary.

## Why Now

Five forces converge in 2026:
1. Seasonal labour pipeline is structurally breaking (not cyclical)
2. RGB-D cameras dropped from $500 → $150, LiDAR from $10k → <$1k (2020–2024)
3. SAIA's €10M raise proves investors will fund this thesis
4. Dutch government infrastructure — HIC venture studio, €450M+ in agri-innovation funding
5. Vertical farming collapse released agri-robotics talent and drove hardware costs down

## Business Model

**Robot-as-a-Service (RaaS)** — monthly subscription, no upfront purchase, 12-month minimum.
One robot (€5,800/month) replaces 3–4 workers — cheaper than the labour it replaces from month one.

## Target Market

- **Beachhead:** Dutch tomato/cucumber/pepper operators, 3–10 ha, Westland/Lansingerland
- **SAM (Netherlands):** €167–223M/year
- **Long-term (Europe):** €1.3–2.5B/year

## Status

Pre-prototype. Simulation-first development in ROS 2 + Gazebo.
The single most important thing to prove: **~450–600 fruit/hr throughput** to replace 3 workers.
