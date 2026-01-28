# Learning Context in Football Possessions with Self-Supervised Transformers

This project explores how sequence models can learn **contextual structure in football possessions** using only event data.

Using StatsBomb open event data, I train a **self-supervised transformer** on possession-level event sequences. The model learns by randomly masking actions and predicting them from surrounding context — similar in spirit to masked language modeling, but applied to football actions.

The goal is not to predict outcomes directly, but to learn **what “makes sense” next within a possession**.

---

## Motivation

Traditional football analytics often evaluate actions in isolation (e.g., xG, pass value).  
However, analysts and coaches reason in **sequences**:

- Was this possession *building toward something*?
- Did the attack break down despite “reasonable-looking” actions?
- Do certain action patterns consistently precede shots?

This project investigates whether a transformer can learn these patterns **without labels**, purely from event sequences.

---

## Dataset

- **Source:** StatsBomb Open Data  
- **Data type:** Event-level football data (passes, carries, dribbles, shots, etc.)
- **Granularity:** Possession-level sequences
- **Teams:** Example analysis shown for FC Barcelona

No proprietary or private data is used.

Instructions to download the data are provided in `data/statsbomb_info.md`.

---

## Method

### 1. Possession Encoding

Each event in a possession is represented as:
- Action type (pass, carry, dribble, shot, etc.)
- Zone / spatial context
- Temporal gap (`deltaT`)
- Additional engineered contextual features

These are embedded and concatenated into a sequence.

---

### 2. Self-Supervised Training (Masked Event Modeling)

During training:
- Random events in a possession are masked
- The transformer is trained to predict the masked actions from surrounding context

This forces the model to learn:
- short-term dependencies (local combinations)
- long-range dependencies (early build-up → later outcomes)

No labels such as goals or success are used.

---

### 3. Why a Transformer?

A transformer is necessary because:
- possessions vary in length
- relevant context may occur many actions earlier
- order and interaction between actions matters

Simple aggregation or Markov-style models cannot capture this structure.

---

## Interpreting the Model

Below are examples of masked-action prediction in different possession types.

**Grey dots:** true action sequence  
**Colored dots:** model’s top-3 predictions when an action is masked

### Shot-ending possession
![Shot possession](visuals/vis_shot_possession.png)

The model assigns higher probability to shot-like actions when the surrounding context resembles attacking build-ups.

---

### Slow / broken possession
![Turnover possession](visuals/vis_turnover_possession.png)

Here, predictions are more uncertain, reflecting ambiguous or stalled sequences.

---

### Dribble-heavy possession
![Dribble possession](visuals/vis_dribble_possession.png)

The model learns that aggressive ball-carrying patterns often precede shots, even when individual events are masked.

---

## Key Takeaway

The transformer is not memorizing actions.

It is learning **contextual expectations**:
> “Given everything that happened before and after, what action would make sense here?”

This is a foundation for analyst-facing tools such as:
- flagging possessions that *look correct but fail*
- identifying recurring possession templates
- comparing possessions by structure rather than surface stats

---

## Limitations

- Uses only open event data (no tracking data)
- No direct evaluation against coaching decisions
- Focuses on representation learning, not deployment

This is an exploratory, learning-oriented project.

---

## Future Directions

- Build analyst-facing summaries on top of learned representations
- Compare possession similarity across teams or matches
- Integrate richer spatial or tracking information

---

## Author

Pratham Sharma  
M.S. in Computer Science  
Interested in applied ML for sports analytics and decision-support systems
