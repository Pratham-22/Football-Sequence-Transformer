StatsBomb Open Data

Overview

This project uses StatsBomb Open Data, a publicly available football event dataset released for research and educational purposes.

The dataset contains event-level match data describing on-ball actions such as passes, carries, dribbles, shots, duels, and fouls, along with spatial and temporal context.

Source:
StatsBomb Open Data GitHub
https://github.com/statsbomb/open-data

No proprietary, private, or club-restricted data is used in this project.

⸻

Data Characteristics
	•	Data type: Event-level football data
	•	Granularity: Sequential possession-level event streams
	•	Spatial information: Event coordinates (x, y) in pitch space
	•	Temporal information: Timestamps and inter-event time (deltaT)
	•	Teams: Example analyses and visualizations focus on FC Barcelona, but the pipeline is team-agnostic
	•	Competitions: Multiple leagues and seasons provided by StatsBomb Open Data

Each match is stored as a JSON file containing a chronological list of events.

⸻

How the Data Is Used in This Project

From the raw event data, we construct:
	•	Possession-level sequences (ordered events within the same possession)
	•	Fixed-length context windows (e.g., 40 consecutive events)
	•	Action tokens (pass, carry, dribble, shot, etc.)
	•	Context features, including:
	•	Zone-based spatial features
	•	Relative movement (delta x, delta y)
	•	Inter-event time (deltaT)

These sequences are then used to train a self-supervised transformer using a masked-event modeling objective.



Download Instructions: 
	1.	Clone the StatsBomb Open Data repository:

```
git clone https://github.com/statsbomb/open-data.git
```
  2.  Navigate to the events directory:
```
cd open-data/data/events
```
  3.  Select the competitions / matches you want to analyze.
	4.	Place the downloaded event files inside this repository under:
```
data/events
```
Licensing & Usage

StatsBomb Open Data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
