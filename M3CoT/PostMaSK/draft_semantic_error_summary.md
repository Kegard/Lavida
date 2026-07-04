# Draft Semantic Error Classification

Records: `M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_conf_r4_seed42_n400/records.jsonl`

Definitions used:

- `local`: the draft error is attributable to a specific local token span or reasoning step, and a minimal local edit could recover the answer.
- `global`: the draft's semantic trajectory is incomplete, unsupported, or misaligned enough that fixing an isolated span is insufficient.

Important convention: changing only the final `\boxed{}` letter is not counted as a local repair unless the preceding CoT already contains a localized answer-bearing mistake that explains the wrong conclusion.

## Summary

| Total | Draft Correct | Draft Wrong | Local | Global |
|---:|---:|---:|---:|---:|
| 400 | 156 | 244 | 113 (0.463) | 131 (0.537) |

## Global Subtypes

| Subtype | Count | Ratio among draft errors |
|---|---:|---:|
| `insufficient_or_incomplete_cot` | 93 | 0.381 |
| `metric_keyword_artifact_without_answer` | 36 | 0.148 |
| `unsupported_reasoning_trajectory` | 2 | 0.008 |

## By Extraction Source

| Extract Source | Total Wrong | Local | Global |
|---|---:|---:|---:|
| `answer_colon` | 1 | 1 | 0 |
| `boxed` | 113 | 111 | 2 |
| `choice_text_keyword` | 12 | 0 | 12 |
| `failed` | 92 | 0 | 92 |
| `letter_token_lower` | 17 | 1 | 16 |
| `letter_token_upper` | 9 | 0 | 9 |

## Topic Breakdown

| Topic | Wrong | Local | Global |
|---|---:|---:|---:|
| `social-science` | 85 | 24 | 61 |
| `natural-science` | 83 | 27 | 56 |
| `language-science` | 20 | 16 | 4 |
| `social-commonsense` | 18 | 10 | 8 |
| `algebra` | 17 | 17 | 0 |
| `geometry` | 11 | 10 | 1 |
| `temporal-commonsense` | 5 | 4 | 1 |
| `physical-commonsense` | 4 | 4 | 0 |
| `theory` | 1 | 1 | 0 |

## Representative Local Examples

- idx=8, id=mathematics-1649, gold=A, pred=C, source=boxed: Look at the following schedule. When does Time Traveler II begin?
  - span: ` According to the schedule, Time Traveler II begins at 12:15 P.M. Therefore, the correct answer is: \boxed{C}`
  - draft: `To determine when Time Traveler II begins, we need to look at the "Begin" column for the corresponding row in the movie schedule.  According to the schedule, Time Traveler II begins at 12:15 P.M.  Therefore, the correct answer is:  \boxed{C}`
- idx=9, id=cognitive-science-264, gold=C, pred=D, source=boxed: What object can be seen in the image?
  - span: `h a small) at the top, resembling the typical features of a cup. .Therefore, the correct answer is: \boxed{D}`
  - draft: `The object in the image appears to be a cup. This deduction is based on the shape shape of the object, which includes a cylindrical body and a handle ( with a small) at the top, resembling the typical features of a cup. .Therefore, the correct answer is:  \boxed{D}`
- idx=11, id=writing-strategies-457, gold=A, pred=B, source=boxed: Based on the time line, which book was published later?
  - span: `er Chair" was published in 1953 and 1953. Therefore, "The Silver Chair" was published later. Answer \boxed{B}`
  - draft: `To determine which book was published later, we need to look at the publication dates in the timeline. The publication shows that "The Silver Chair" was published in 1953 and 1953. Therefore, "The Silver Chair" was published later.  Answer \boxed{B}`
- idx=20, id=mathematics-891, gold=B, pred=D, source=boxed: Examine the given choices given in the text carefully and identify the one that best solves the Geometry problem in the image.
  - span: `To determine the correct answer, let's analyze the options provided: - Option A: 6 meters. - Option D: 10 meters. The correct height from the geometry problem is 10 meters. Therefore, the correct answer is: \boxed{D`
  - draft: `To determine the correct answer, let's analyze the options provided:  - Option A: 6 meters. - Option D: 10 meters.  The correct height from the geometry problem is 10 meters.  Therefore, the correct answer is:  \boxed{D`
- idx=22, id=biology-1745, gold=C, pred=B, source=boxed: Determine the scientific name of the organism featured in the main image, using the provided options within the image.
  - span: `ws a green frog, which corresponds to option (B) Atelopus zeteki. Therefore, the correct answer is: \boxed{B}`
  - draft: `To determine the scientific name of the organism featured in the main image, let's analyze the options provided:  1. The main image shows a green frog, which corresponds to option (B) Atelopus zeteki.  Therefore, the correct answer is:  \boxed{B}`
- idx=24, id=physical-commonsense-889, gold=B, pred=A, source=boxed: In the image, you can observe numerous glass with nearly identical quality liquid.
  - span: `reezing point than Container G, so it will solidify more quickly. Therefore, the correct answer is: \boxed{A}`
  - draft: `To determine which liquid will undergo the process of solidification more quickly when placed in the consider the freezing points of the liquids.. the liquid in Container B has a lower freezing point than Container G, so it will solidify more quickly.  Therefore, the correct answer is:  \boxed{A}`
- idx=26, id=physical-commonsense-578, gold=D, pred=B, source=boxed: There are several items listed in the picture.Additionally, we should consider the room temperature to be 25 °C at this moment.
  - span: `ures of all liquid liquids are lower than room temperature (25°C. Therefore, the correct answer is: \boxed{B}`
  - draft: `To determine which container will display droplets on the inner wall first, we need to consider the temperature of the liquids inside the container relative to the room temperature. Since the temperatures of all liquid liquids are lower than room temperature (25°C.  Therefore, the correct answer is:  \boxed{B}`
- idx=27, id=biology-2108, gold=B, pred=D, source=boxed: Using the options displayed in the image, ascertain the scientific name of the organism shown in the primary image.
  - span: ` organism shown in the primary image is: D. Falco novaeseelandiae Therefore, the correct answer is: \boxed{D}`
  - draft: `The primary image shows a bird that appears to be an owl, specifically a barn owl. Based on the options provided, the scientific name for the organism shown in the primary image is:  D. Falco novaeseelandiae  Therefore, the correct answer is:  \boxed{D}`
- idx=30, id=social-commonsense-1148, gold=D, pred=A, source=letter_token_lower: What is happening in the given picture?
  - span: `correct answer, let's analyze the image step by step: 1. **Identify the activity**: The image shows a person performing on a stage, with other individuals around them. The. The individuals are holding musical instruments`
  - draft: `To determine the correct answer, let's analyze the image step by step:  1. **Identify the activity**: The image shows a person performing on a stage, with other individuals around them. The. The individuals are holding musical instruments and appear to be part of the performance.  2`
- idx=32, id=physics-1462, gold=C, pred=A, source=boxed: Select the best answer.
  - span: `are typically hard. While not share other2, blue or being sticky. Therefore, the correct answer is: \boxed{A}`
  - draft: `To determine the common property among the three objects -camel corn, bubble gum, and tape - let's analyze each option:  1. Hard three objects are typically hard. While not share other2, blue or being sticky.  Therefore, the correct answer is:  \boxed{A}`
- idx=36, id=social-commonsense-1648, gold=C, pred=B, source=boxed: What might the woman at the casual dining restaurant be doing?
  - span: `directed towards the camera, that suggests she is taking a photo. Therefore, the correct answer is: \boxed{B}`
  - draft: `The woman at the casual dining restaurant appears to be taking a selfie. This is indicated by her way she is holding her hand near her face, possibly a cell phone, and her focused directed towards the camera, that suggests she is taking a photo. Therefore, the correct answer is:  \boxed{B}`
- idx=39, id=mathematics-86, gold=B, pred=A, source=boxed: This Algebra question has multiple options. Which one should I choose?
  - span: `lem statement, if we assume that each student in the which is 30. Therefore, the correct choice is: \boxed{A}`
  - draft: `To determine the correct answer, let's analyze the options provided:  1. The D option states 30. 2. According to the problem statement, if we assume that each student in the which is 30.  Therefore, the correct choice is:   \boxed{A}`

## Representative Global Examples

- idx=0, id=economics-592, gold=D, pred=FAILED, subtype=insufficient_or_incomplete_cot: The image provides an overview of the economic situation in Country 1, Country 3 and Country 2.
  - draft: `To determine the average hourly wages for Country 1, Country 3, and Country 2 for the year 2023, we need to look at the weekly per capita capita wage for each country and then convert it to an hourly wage by dividing the average weekly working hours by 40 hours.`
- idx=2, id=physical-commonsense-1157, gold=C, pred=B, subtype=metric_keyword_artifact_without_answer: In the picture, you can see multiple purchased items. Additionally, we are operating under the assumption that the current water temperature stands at 4 °C.
  - draft: `To determine the number of objects that seem to be buoyant on the water, we need to consider the density of each object and compare it to the density of water at the temperature of water (4°C), which is 1000 kg/m³³). An object is buoyant if its density is`
- idx=4, id=economics-1055, gold=B, pred=FAILED, subtype=insufficient_or_incomplete_cot: This is the image about the economics status for several countries.
  - draft: `To determine which country has the highest GDP per capita in 2016, we need to compare the GDP per capita values for Country 4, Country 3, and Country 2 in that year year.  1. **Country 4 in 2016:**    - GDP`
- idx=6, id=geography-3840, gold=A, pred=C, subtype=metric_keyword_artifact_without_answer: The photograph reveals a terrain chart depicting a particular area. The photograph reveals a terrain chart depicting a particular area. The figure contains a contour map and a directional marker.
  - draft: `To determine the elevations of points C and B, we need to look at the color lines on the contour and use the color bar provided in the image.  1. **Point C**:    - Point C is located in a dark red region.    - According to the color bar, dark red`
- idx=7, id=physical-commonsense-1105, gold=C, pred=D, subtype=metric_keyword_artifact_without_answer: There are several purchased items listed in the picture. Additionally, we are operating under the assumption that the current water temperature stands at 4 °C.
  - draft: `To determine the number of objects that seem to be buoyant on water, we need to compare the density of each object to the density of water at 4°C, which is 1000 kg/m³. An object will be buoyant if its density is less than the density of water. Let`
- idx=10, id=economics-76, gold=B, pred=FAILED, subtype=insufficient_or_incomplete_cot: The image provides an overview of the economic situation in some countries.
  - draft: `To determine the trade balances of Country 2 and Country 3 for the year 2022, we need to calculate the export balance for each country and subtracting the import value from the export value.  1. 2022 Country 2: - - Export Value: $1`
- idx=12, id=geography-3391, gold=E, pred=C, subtype=metric_keyword_artifact_without_answer: The picture illustrates how the depths are distributed within a particular ocean. Within the image, you can see a contour map and a directional marker.
  - draft: `To determine the depths at points C and A, we need to refer to the color scale on the right side of the image, as the color at each point indicates the depth.  1. **Point C**:    - Point C is located in a dark blue region.    - According to the`
- idx=13, id=economics-1322, gold=C, pred=FAILED, subtype=insufficient_or_incomplete_cot: This is the image about the economics status for several countries.
  - draft: `To determine which country has the highest GDP per capita in 2021, we need to compare Country 3, Country 5, and Country 2 for that year year.  1. **Country 3 in 2021:**    - GDP: 13.9`
- idx=14, id=economics-706, gold=A, pred=FAILED, subtype=insufficient_or_incomplete_cot: Here is a image showcasing the economic status of Country 5, Country 1 and Country 2.
  - draft: `To determine the average hourly wages figures for Country 5, Country 1, and Country 2 in 2023, we need to look at the data provided in the table for that year.  1. **Country 5 in 2023:**    - Weekly Per Hour`
- idx=16, id=social-commonsense-1144, gold=B, pred=FAILED, subtype=insufficient_or_incomplete_cot: What is happening in the picture?
  - draft: `To determine what is happening in the picture, let's analyze the details:  1.`
- idx=17, id=physical-commonsense-1071, gold=C, pred=FAILED, subtype=insufficient_or_incomplete_cot: In the image, you can observe numerous purchased items. Additionally, we assume that the water temperature is 4 ° C at this time.
  - draft: `To determine how many items will submerge underwater, we need to compare the density of each object to the density of water at 4°C. The density of water at`
- idx=19, id=geography-520, gold=A, pred=FAILED, subtype=insufficient_or_incomplete_cot: This graph displays data on both precipitation and temperature within a particular geographic area.
  - draft: `To characterize the climate seen in the photo, let's analyze the data presented in the graph:  graph`
