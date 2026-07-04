# Draft Text Error Source Summary

This file analyzes only `draft_text`, before PostMaSK refinement.

Definitions:

- `global_explicit_wrong_answer`: the draft explicitly outputs a wrong option through `Answer:`, `\boxed{}`, `[Answer]`, or `(A)`-style answer syntax.
- `local_keyword_extraction_error`: the draft has no explicit final answer, and the metric picks a wrong option because of a local keyword, option text, or standalone option letter in the draft.
- `local_incomplete_or_no_answer`: the draft has no extractable answer at all, usually because it stops in the middle of reasoning.

These labels are automatic proxies. `global_explicit_wrong_answer` is the strongest signal that the draft's overall conclusion is wrong; `local_keyword_extraction_error` is the strongest signal that a local phrase or keyword caused the judged answer.

## Experiment Table

| Experiment | N | Draft Acc | Draft Wrong | Global Explicit Wrong | Local Keyword | Incomplete/No Answer |
|---|---:|---:|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_cachedconf_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_fullconf_r4_seed42_n400` | 2 | 0.500 | 1 | 0 (0.000) | 0 (0.000) | 1 (1.000) |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_histconf_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_kldiv_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_random_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | 400 | 0.390 | 244 | 114 (0.467) | 38 (0.156) | 92 (0.377) |
| `postmask_sr0p5_d32_p0_seed42_n400` | 400 | 0.407 | 237 | 107 (0.451) | 41 (0.173) | 89 (0.376) |

## Extract Source Table

| Experiment | Explicit Sources | Choice Text Keyword | Letter Token | Failed |
|---|---:|---:|---:|---:|
| `postmask_sr0p5_d16_p16_cachedconf_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_cachedconf_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_conf_r2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_conf_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_fullconf_r4_seed42_n400` | 1 | 0 | 0 | 1 |
| `postmask_sr0p5_d16_p16_histconf_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_histconf_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_histconf_r4_seed42_n400_new_log` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_kldiv_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_kldiv_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_kldiv_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_laststepconf_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_laststepconf_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_meanafterfill_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_meanafterfill_r4_seed42_n400_new_log` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_proposalconf_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_rand_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_random_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_random_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_topkmargin_fixed16_refill1_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d16_p16_topkmargin_r4_seed42_n400` | 243 | 31 | 34 | 92 |
| `postmask_sr0p5_d32_p0_seed42_n400` | 236 | 40 | 35 | 89 |

## Representative Examples

### `postmask_sr0p5_d16_p16_conf_r4_seed42_n400`

#### `global_explicit_wrong_answer`

- idx=8, id=mathematics-1649, gold=A, pred=C, source=boxed: Look at the following schedule. When does Time Traveler II begin?  
  `To determine when Time Traveler II begins, we need to look at the "Begin" column for the corresponding row in the movie schedule.  According to the schedule, Time Traveler II begins at 12:15 P.M.  Therefore, the correct answer is:  \boxed{C}`
- idx=9, id=cognitive-science-264, gold=C, pred=D, source=boxed: What object can be seen in the image?  
  `The object in the image appears to be a cup. This deduction is based on the shape shape of the object, which includes a cylindrical body and a handle ( with a small) at the top, resembling the typical features of a cup. .Therefore, the correct answer is:  \box`
- idx=11, id=writing-strategies-457, gold=A, pred=B, source=boxed: Based on the time line, which book was published later?  
  `To determine which book was published later, we need to look at the publication dates in the timeline. The publication shows that "The Silver Chair" was published in 1953 and 1953. Therefore, "The Silver Chair" was published later.  Answer \boxed{B}`
- idx=20, id=mathematics-891, gold=B, pred=D, source=boxed: Examine the given choices given in the text carefully and identify the one that best solves the Geometry problem in the image.  
  `To determine the correct answer, let's analyze the options provided:  - Option A: 6 meters. - Option D: 10 meters.  The correct height from the geometry problem is 10 meters.  Therefore, the correct answer is:  \boxed{D`
- idx=22, id=biology-1745, gold=C, pred=B, source=boxed: Determine the scientific name of the organism featured in the main image, using the provided options within the image.  
  `To determine the scientific name of the organism featured in the main image, let's analyze the options provided:  1. The main image shows a green frog, which corresponds to option (B) Atelopus zeteki.  Therefore, the correct answer is:  \boxed{B}`

#### `local_keyword_extraction_error`

- idx=2, id=physical-commonsense-1157, gold=C, pred=B, source=choice_text_keyword: In the picture, you can see multiple purchased items. Additionally, we are operating under the assumption that the current water temperature stands at 4 °C.  
  `To determine the number of objects that seem to be buoyant on the water, we need to consider the density of each object and compare it to the density of water at the temperature of water (4°C), which is 1000 kg/m³³). An object is buoyant if its density is`
- idx=6, id=geography-3840, gold=A, pred=C, source=letter_token_upper: The photograph reveals a terrain chart depicting a particular area. The photograph reveals a terrain chart depicting a particular area. The figure contains a contour map and a directional marker.  
  `To determine the elevations of points C and B, we need to look at the color lines on the contour and use the color bar provided in the image.  1. **Point C**:    - Point C is located in a dark red region.    - According to the color bar, dark red`
- idx=7, id=physical-commonsense-1105, gold=C, pred=D, source=choice_text_keyword: There are several purchased items listed in the picture. Additionally, we are operating under the assumption that the current water temperature stands at 4 °C.  
  `To determine the number of objects that seem to be buoyant on water, we need to compare the density of each object to the density of water at 4°C, which is 1000 kg/m³. An object will be buoyant if its density is less than the density of water. Let`
- idx=12, id=geography-3391, gold=E, pred=C, source=letter_token_upper: The picture illustrates how the depths are distributed within a particular ocean. Within the image, you can see a contour map and a directional marker.  
  `To determine the depths at points C and A, we need to refer to the color scale on the right side of the image, as the color at each point indicates the depth.  1. **Point C**:    - Point C is located in a dark blue region.    - According to the`
- idx=29, id=geography-3369, gold=A, pred=C, source=letter_token_upper: This image depicts the distribution of ocean depths in a specific ocean. Within the image, you can see a contour map.  
  `To determine the depths at points C and B, we need to refer to the color scale on the right side of the image, as the color bar indicates the depth values in kilometers.  1. **Point C**:    - Point C is located in a dark blue region.    - According to`

#### `local_incomplete_or_no_answer`

- idx=0, id=economics-592, gold=D, pred=FAILED, source=failed: The image provides an overview of the economic situation in Country 1, Country 3 and Country 2.  
  `To determine the average hourly wages for Country 1, Country 3, and Country 2 for the year 2023, we need to look at the weekly per capita capita wage for each country and then convert it to an hourly wage by dividing the average weekly working hours by 40 hour`
- idx=4, id=economics-1055, gold=B, pred=FAILED, source=failed: This is the image about the economics status for several countries.  
  `To determine which country has the highest GDP per capita in 2016, we need to compare the GDP per capita values for Country 4, Country 3, and Country 2 in that year year.  1. **Country 4 in 2016:**    - GDP`
- idx=10, id=economics-76, gold=B, pred=FAILED, source=failed: The image provides an overview of the economic situation in some countries.  
  `To determine the trade balances of Country 2 and Country 3 for the year 2022, we need to calculate the export balance for each country and subtracting the import value from the export value.  1. 2022 Country 2: - - Export Value: $1`
- idx=13, id=economics-1322, gold=C, pred=FAILED, source=failed: This is the image about the economics status for several countries.  
  `To determine which country has the highest GDP per capita in 2021, we need to compare Country 3, Country 5, and Country 2 for that year year.  1. **Country 3 in 2021:**    - GDP: 13.9`
- idx=14, id=economics-706, gold=A, pred=FAILED, source=failed: Here is a image showcasing the economic status of Country 5, Country 1 and Country 2.  
  `To determine the average hourly wages figures for Country 5, Country 1, and Country 2 in 2023, we need to look at the data provided in the table for that year.  1. **Country 5 in 2023:**    - Weekly Per Hour`
