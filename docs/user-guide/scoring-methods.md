# Scoring Methods

Climb Analyzer supports three scoring algorithms to rate climb difficulty. Each has different strengths.

## Quick Comparison

| Method | Best For | Complexity | Notes |
|--------|----------|------------|-------|
| Basic | Quick comparisons | Simple | distance × grade |
| FIETS | European-style climbs | Medium | Rewards steep, high climbs |
| PDI | Comprehensive analysis | Complex | Most accurate difficulty |

## Basic Score

The simplest scoring method: **distance × average grade**.

### Formula

```
Basic Score = distance (meters) × average grade (%)
```

### Examples

| Climb | Distance | Grade | Basic Score |
|-------|----------|-------|-------------|
| Short & Steep | 2 km | 10% | 20,000 |
| Long & Gentle | 10 km | 3% | 30,000 |
| Medium | 5 km | 5% | 25,000 |

### Interpretation

- Higher score = more challenging overall
- Doesn't distinguish between long/gentle vs short/steep
- Good for quick comparisons

### When to Use

- Quick filtering of climbs
- When you want a simple metric
- Comparing climbs of similar character

---

## FIETS Index

Developed by Dutch cycling magazine *Fiets*. Rewards steeper climbs and high-altitude finishes.

### Formula

```
FIETS = (H² / D × 10) + max(0, (T - 1000) / 1000)
```

Where:
- **H** = Elevation gain (meters)
- **D** = Distance (meters)
- **T** = Summit elevation (meters)

### Breakdown

The FIETS score has two components:

1. **Gradient Factor**: `H² / D × 10`
   - Squares the elevation gain, heavily rewarding steep climbs
   - A 10% grade scores much higher than a 5% grade

2. **Altitude Bonus**: `max(0, (T - 1000) / 1000)`
   - Adds points for climbs finishing above 1000m
   - Every 1000m above 1000m adds 1 point

### Examples

| Climb | H (m) | D (m) | T (m) | FIETS |
|-------|-------|-------|-------|-------|
| Alpe d'Huez | 1120 | 13800 | 1850 | 9.9 |
| Mont Ventoux | 1617 | 21500 | 1912 | 13.1 |
| Col du Tourmalet | 1404 | 17200 | 2115 | 12.6 |

### Interpretation

| FIETS Score | Difficulty |
|-------------|------------|
| < 2 | Easy |
| 2-4 | Moderate |
| 4-6 | Challenging |
| 6-8 | Difficult |
| 8-10 | Very Difficult |
| > 10 | Extreme |

### When to Use

- Comparing European mountain climbs
- When altitude matters
- Traditional cycling analysis

---

## PDI (PJAMM Difficulty Index)

The most sophisticated scoring method, developed by [PJAMM Cycling](https://pjammcycling.com/blog/50.pjamm-difficulty-index).

### Key Innovations

1. **Total Work, Not Just Elevation**
   - Accounts for work against friction on flat sections
   - More accurate representation of effort

2. **Descent Penalties**
   - Flat descents in a climb allow recovery
   - PDI penalizes these "rest" sections

3. **Physics-Based**
   - Based on power required to climb
   - Considers rolling resistance and air resistance

### Formula Components

```
PDI = Work_climbing + Work_resistance - Recovery_bonus
```

Where:
- **Work_climbing** = Energy to overcome gravity
- **Work_resistance** = Energy for rolling/air resistance
- **Recovery_bonus** = Deduction for descent sections

### Comparison with Basic/FIETS

| Scenario | Basic | FIETS | PDI |
|----------|-------|-------|-----|
| Steady 5% grade | Medium | Medium | Medium |
| Variable grade (3-10%) | Same | Similar | Higher (harder) |
| Climb with descent | Same | Same | Lower (recovery) |
| High altitude finish | Same | Higher | Similar |

### When to Use

- Most accurate difficulty assessment
- Comparing climbs with varied profiles
- When descent recovery matters

---

## Cycling Categories

Based on scores, climbs are categorized like European cycling races:

| Category | Description | Typical Characteristics |
|----------|-------------|-------------------------|
| **HC** | Hors Catégorie | Extreme climbs, Tour de France summit finishes |
| **Cat 1** | Very Difficult | Major mountain passes |
| **Cat 2** | Difficult | Significant climbs |
| **Cat 3** | Moderate | Rolling terrain, longer hills |
| **Cat 4** | Easy | Short climbs, gentle grades |

### Category Thresholds

Thresholds vary by scoring method. Approximate Basic Score ranges:

| Category | Basic Score |
|----------|-------------|
| HC | > 80,000 |
| Cat 1 | 50,000 - 80,000 |
| Cat 2 | 30,000 - 50,000 |
| Cat 3 | 15,000 - 30,000 |
| Cat 4 | 6,000 - 15,000 |

---

## Choosing a Scoring Method

### Use Basic Score When:
- You want a quick overview
- Comparing many climbs rapidly
- Simplicity is preferred

### Use FIETS When:
- Analyzing European mountain climbs
- Altitude is important
- Following traditional cycling metrics

### Use PDI When:
- You want the most accurate difficulty rating
- Comparing climbs with varied profiles
- Descent recovery matters to your analysis

## CLI Usage

```bash
# Basic score (default)
./climb-analyzer -r "Vermont" -t basic

# FIETS index
./climb-analyzer -r "Switzerland" -t fiets -u metric

# PDI
./climb-analyzer -r "Colorado" -t pdi
```

## Output Fields

All three scores can be computed, but the selected score type determines:
- The default sort order in output
- The score used for category assignment
- The primary score displayed in GUI

---

Next: [Output Format](output-format.md)
