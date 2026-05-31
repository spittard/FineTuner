# SME plugging assessment (heuristic)

Rule-based flags for rows where a **subject-matter expert might disagree** with the top match or score story. **Not** ground truth—use for triage.

Regenerate after full `plugging_matches.json`:

`python scripts/sme_assess_plugging_matches.py --fresh --run-label <label>`



---

## Run `plan-no-geo-exact-20260430` — 3217 records, batch_size=200

_Generated: 2026-05-01T12:45:06.595860+00:00_

### Batch 1 (records 0..199)

- **In batch:** 200 | **Flagged:** 87 | **Flag rate:** 43.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7879015 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Servant Keeper, LLC | SKL |
| 7879342 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | NWACUHO - Northwest Association of College & University | NW Association of College & University Housing Officers |
| 7880903 | `TOP5_TIE_CLUSTER` | 0.9746413510344029 | Corporation of Hamilton | Hamilton Enterprises |
| 7880906 | `TOP5_TIE_CLUSTER` | 0.9591999999999999 | Corporation Of Hamilton | HAMILTON GROUP |
| 7893800 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Perfect Planning Events | PPE |
| 7894031 | `LOW_CONFIDENCE_TOP1` | 0.5402097833803965 | CU Cooperative Systems, Inc. dba CO-OP Solutions | Co-op Solutions |
| 7895819 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Lanthrop Gpm | GPM Life |
| 7896046 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9023634006030621 | Emergency Medical Services/DSHS | Emergency Medical Service |
| 7913800 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | HB | H Beck |
| 7918771 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | X-Camp Academy | XCA |
| 7918803 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Wisconsin Evangelical Lutheran Church | WELS -Wisconsin Evangelical Lutheran Synod |
| 7918805 | `LOW_CONFIDENCE_TOP1` | 0.53 | Soluz LLC | Soluz LLC |
| 7918806 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | She's Still Missing | SSM |
| 7918809 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5182537969675931 | Quadcode | Quad |
| 7918830 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7677402474691148 | Greenlight Tour & Travel, Inc.  d/b/a GL Travel | GL Travel |
| 7918831 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7677402474691148 | Greenlight Tour & Travel, Inc.  d/b/a GL Travel | GL Travel |
| 7918832 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7677402474691148 | Greenlight Tour & Travel, Inc.  d/b/a GL Travel | GL Travel |
| 7918896 | `LOW_CONFIDENCE_TOP1` | 0.53 | SNAMO - AVEO Table + Bar | SNAMO - AVEO Table + Bar |
| 7918897 | `LOW_CONFIDENCE_TOP1` | 0.53 | SNAMO - AVEO Table + Bar | SNAMO - AVEO Table + Bar |
| 7918909 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | North End Teleservices | NET |
| 7918915 | `LOW_CONFIDENCE_TOP1` | 0.518576575074506 | Pimcore | Pimco |
| 7918954 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.851537670494756 | Rensenhouse Industrial Solutions KC | KC Industrial Council |
| 7919032 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8294938886144573 | Kingdom Alliance Repour and Relaxation | Kingdom Alliance Global Ministries |
| 7919049 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | DLM Governance July 2026 | DMS Governance |
| 7919298 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Superior Environmental Solutions | SES |
| 7919301 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7621648355120627 | System of Care Convening | Care Medical |
| 7919307 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | RHM Meeting | Personal Meeting |
| 7919311 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SAFTE-FAST User Conference | Fast Enterprise |
| 7919314 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8236609637262011 | Ontario Skate League Championships | Ontario Cup Provincial Championships |
| 7919320 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7753016121343691 | Pyramid Systems Annual Excellence Awards | Pyramid Systems Inc. |
| 7919333 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7649896967635748 | RDB Hospitality -Rick Marino Group-  - Toronto | RDB Hospitality |
| 7919334 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | September Group | September Events |
| 7919436 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7690300259927076 | Niantic-2026 Jun-Staff/Production Room Block | Production Company Room Block |
| 7919442 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7919444 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | SAEOPP 2028 Annual Conference | SAEOPP 2024 Training Conference |
| 7919451 | `TOP5_TIE_CLUSTER` | 0.999 | The Washington Institute | The Washington Institute |
| 7919455 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | NOSAC Spring Meetin | NSM |
| 7919459 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8401218043718859 | Stumm Executive Sales Trip | Inside Sales Incentive Trip |
| 7919525 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | 4Minds / Padrao - ID 180299 - INNOVATION SUMMIT - Orlan | Innovation Summit |
| 7919647 | `LOW_CONFIDENCE_TOP1` | 0.53 | CRLA 2026 Retreat | CRC Women's Retreat |
| 7919662 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.45620294128078237 | WynnLuxe Curations | Wynn Solutions |
| 7919672 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Pearl Behavioral Health Services | PBHS |
| 7919683 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Annual Meeting of the Joint Society ASSCT | Annual Meeting |
| 7919688 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7828375722233406 | 2027 SCABB ANNUAL MEETING | SCABB Management Meeting |
| 7919701 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sigma Phi Society | SPS |
| 7919704 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | OALA Training | Hands on Training |
| 7919706 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | SBCI Conferance | THE CONFERANCE BOARD |
| 7919716 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8019751210024825 | Cruise SCJ Feb 2026 | Cruise Rooms Feb 2018 |
| 7919755 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Schattastrum Industries | Omni Industries |
| 7919762 | `LOW_CONFIDENCE_TOP1` | 0.5482908104460993 | RFP 93: TD Bank-2026 May-Spring Management Conference ( | 2026 Fall HR Management Conference |
| 7919765 | `LOW_CONFIDENCE_TOP1` | 0.53 | Snap Inc | Snap Incorporated, LLC |
| 7919771 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.780001685195874 | SAT Mexico | TRAXION CEO Summit 2026 | SAT Mexico | Grupo INCENTIVO MULTI 2026 |
| 7919787 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7838146671219304 | Southern Ute Indian Tribe Growth Fund | Southern Ute Tribe - Regulatory Division |
| 7919804 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spring 2026 District Meeting | Big Spring High School |
| 7919805 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Resort Lovers International | RLI |
| 7919809 | `LOW_CONFIDENCE_TOP1` | 0.5231163200960212 | Speakpreneur | Couplepreneurs |
| 7919828 | `LOW_CONFIDENCE_TOP1` | 0.53 | Yahoo Holdings Inc | Yahoo Holdings, Incorporated |
| 7919829 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | PACS Healthcare Elevated | PHE |
| 7919832 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Virtual Cantina Network | VCN |
| 7919833 | `LOW_CONFIDENCE_TOP1` | 0.53 | Varian Inc. | Varian |
| 7919836 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7801813116316436 | Southern Power Company | Purchasing Power |
| 7919841 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Niagara TEM Summit | NTS |
| 7919855 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8241939626700077 | Southwest Mississippi Community College Band | Southwest MS Community College Stageband |
| 7919863 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Restaurant Strategy | Restaurant 365 |
| 7919874 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7758479326963426 | SWM/Pharmaceutical Investigator Meeting/San Francisco | Pharmaceutical Investigator |
| 7919889 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8116474144365692 | OhioHealth Grant Family Medicine | OhioHealth Grant Bone and Joint Center |
| 7919895 | `LOW_CONFIDENCE_TOP1` | 0.53 | Wiz.io | Wiz IO |
| 7919896 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5136861656268283 | NUTESA | Nutec |
| 7919907 | `LOW_CONFIDENCE_TOP1` | 0.53796575881722 | Wayz / Embracon – Orlando | QuestCon Orlando |
| 7919911 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | VistaJet Ltd. | VistaJet Inc. |
| 7919912 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Portside Realty, LLC | Paramount Realty USA Llc. |
| 7919914 | `LOW_CONFIDENCE_TOP1` | 0.53 | VeloSano | VeloSano |
| 7919915 | `LOW_CONFIDENCE_TOP1` | 0.53 | Rinderknecht Associates | Rinderknecht |
| 7919963 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5027287101274837 | Waitwhile | WaitWhat |
| 7919966 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Shine The Light Conference | SLC |
| 7919971 | `LOW_CONFIDENCE_TOP1` | 0.5293656871083811 | RSS Travel, Luis Soto Party, May 05, Orlando | RSS Travel |
| 7919976 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Western Weather Group | Air Methods Western Region |
| 7919977 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.505827805418814 | WP2030 | W20 |
| 7919997 | `LOW_CONFIDENCE_TOP1` | 0.53 | Sexton Group Sales Training | Sales Training |
| 7920012 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Whelan Mellen | Mellen Events |
| 7920014 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8373667039000082 | Pro-ven/128971/Miami | Pro Travel Miami |
| 7920015 | `LOW_CONFIDENCE_TOP1` | 0.5028394949871257 | Oceandusk/World Cup 2026/3-4 star/21 June/Atlanta | Oceandusk Travel |
| 7920016 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8536374544328371 | Pro-ven/128970/Miami | Pro Travel Miami |
| 7920017 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7849263201072063 | Put Progress First | Progress Now |
| 7920022 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pet Business Marketing Ltd | Advantage Business Marketing |
| 7920025 | `LOW_CONFIDENCE_TOP1` | 0.53 | PAD Enterprises | Alpha Enterprises |
| 7920026 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Omega - Annual Client Appreciation | Alpha Omega Publishers |

### Batch 2 (records 200..399)

- **In batch:** 200 | **Flagged:** 94 | **Flag rate:** 47.0%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7920037 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7890975476307462 | Rizes Grupo Senosiain - Chicago | GRUPO 24 Horas Chicago |
| 7920042 | `LOW_CONFIDENCE_TOP1` | 0.53 | SAMPS | SAM Group |
| 7920059 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | RPAG | R P Alpha Group |
| 7920066 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | The Washington Tattoo, Inc. | WTI |
| 7920072 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920114 | `LOW_CONFIDENCE_TOP1` | 0.5489362569694503 | Safti First | SAE Institute |
| 7920120 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | RSC MECHANICAL, INC. | RMI |
| 7920124 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | RG Events/Function-4 | RG Events |
| 7920128 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Whitmore Manufacturing LLC | Rim Manufacturing, LLC |
| 7920139 | `LOW_CONFIDENCE_TOP1` | 0.4886740998131258 | Oceandusk/World Cup 2026/3-4 Star/16 June/Miami | LINKS 2023 Miami |
| 7920141 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Rosemore Capital Strategies LLC | RCSL |
| 7920150 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Omni Invictus LLC | OIL |
| 7920166 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920167 | `LOW_CONFIDENCE_TOP1` | 0.5350068662069274 | W2M/ICARION CT/NY/WA. Roadshow 40-50 PAX. OCTUBRE 2026/ | 2022 Roadshow New York |
| 7920176 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8282191312817655 | Penn Medicine 2nd Annual HR Summit | Penn Medicine |
| 7920187 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SJCSH Online Bookings | Online Bookings -SFOBG |
| 7920189 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920190 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7420223728226 | Senosian DDW Chicago May 2026 | 247 TravelPro - May 2026 -  Chicago |
| 7920192 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7913761922902247 | U GLOW | AHERN Marzo 2026 | | Enso Media | ALLERGAN MARZO 2026 | |
| 7920196 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.798176422562618 | Red Carpet Award Ceremony | VIP Red Carpet Events |
| 7920198 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ServiceNow, Inc. | ServiceNow, Inc. |
| 7920202 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Pacific West Builders | Pacific Steel |
| 7920203 | `LOW_CONFIDENCE_TOP1` | 0.53 | UL LLC | UL |
| 7920206 | `LOW_CONFIDENCE_TOP1` | 0.53 | Watco Companies | Watco Companies |
| 7920219 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7870189416141276 | SAT Mexico | Pinguim 2026 | Hilton Reforma | G2 | Are IM | Showheroes Junio 2026 | Hilton Reforma |
| 7920220 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7945900249457364 | SAT Mexico | Pinguim 2026 | Hilton Reforma | G1 | Are IM | Showheroes Junio 2026 | Hilton Reforma |
| 7920221 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Western Orchard Group | WOG |
| 7920346 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920373 | `TOP5_TIE_CLUSTER` | 0.6790056870745864 | ONE VISION TALMA GROUP | Cancun | Jan 2026 | One Vision |
| 7920422 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ACIU North America 2025 National Convention | Universal North America |
| 7920475 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | American Ramallah Club of DC | American Ramallah Club NY |
| 7920490 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Under Armour Championship | UAC |
| 7920494 | `LOW_CONFIDENCE_TOP1` | 0.5126914423685462 | DFWAN - 2024/25 Solicitation (Maria Montes) | 2026 AAOS Affiliates |
| 7920497 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Special Event for Suraj Pullapantula | Special Event for Adrian Ruiz |
| 7920500 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8718200424698687 | RSS Travel, Shak Larson Group, San Jose, Jun 07 | RSS Travel, Shak Larson Group, Mexico City, Aug 23 |
| 7920501 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7416278841559832 | WSHA Symposium 2027 | WSHA |
| 7920502 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5130305649205911 | WOCON | WOCAN |
| 7920503 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8514236292616923 | RSS Travel, Shak Larson Group, Dallas, Jun 20 | RSS Travel, Shak Larson Group, Mexico City, Aug 23 |
| 7920510 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Wahlstrom / Gillis | Gillis |
| 7920514 | `LOW_CONFIDENCE_TOP1` | 0.53 | Wise Snacks | Wise Snacks |
| 7920518 | `LOW_CONFIDENCE_TOP1` | 0.53 | Procore Technologies, Inc. - Carpinteria, CA | ProCon Technologies Inc. |
| 7920521 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Winograd Group | Hope Winograd |
| 7920527 | `LOW_CONFIDENCE_TOP1` | 0.53 | Printing Packaging & Production Workers Union of North  | Packaging Distributors of America |
| 7920533 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | VT3 Enterprises | VT3 Training Squadron |
| 7920552 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 6th Air Naval Gunfire Liaison Co | 6th Air Naval Gunfire Liaison Co |
| 7920581 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026-Gunslinger Ohio State | Ohio National |
| 7920634 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Pastor's Retirement Celebration | PRC |
| 7920637 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8038299292585009 | Peninsula Recreation Group | Department of Recreation |
| 7920646 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Silk Road Production | SRP |
| 7920654 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | San Lorenzo Unified School District | San Lorenzo Unified School District |
| 7920661 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | VIEWS FC | Cap FC United |
| 7920662 | `LOW_CONFIDENCE_TOP1` | 0.53 | personal on behalf of UPIS | UPIS |
| 7920678 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920696 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Power Stream Fitness | Fluid Power Resource |
| 7920701 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Savvy Church Group | SCG |
| 7920705 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8607441781648565 | Rise Sports Group | ESPN Rise |
| 7920714 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Waugh/Mayer | Waugh & Co, Inc |
| 7920716 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7909977626945273 | Victory Christian Academy Prom | Victory Christian School |
| 7920717 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Organizational Behavior Camp | OBC |
| 7920734 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | PLI Pastoral Leadership Institute | Academy of Religious Leadership |
| 7920753 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8112040590017129 | Yasamin and Justin Room Block | JEREMY SHULL ROOM BLOCK |
| 7920763 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7920775 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7842739033958597 | Orange County GOP (Republican Party) | Florida Orange County Democratic Party |
| 7920790 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8404145864971385 | World Via Travel Network | Travel Quest Network |
| 7920811 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7884211052561401 | WEINIG Group USA | Michael Weinig Inc. |
| 7920823 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7630171610358358 | Savannah Georgia Air National Guard | US Airforce Air National Guard |
| 7920825 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 21st Annual Government-to-Government Tribal Consultatio | Sycuan Tribal Government Office |
| 7920839 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7830900143473769 | SiteSearch Travel Services | SiteSearch |
| 7920848 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7667218304364674 | Silicon Valley Comic Con, LLC | Educare for Silicon Valley |
| 7920858 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Rosen Injury Law | RIL |
| 7920874 | `TOP5_TIE_CLUSTER` | 0.999 | The Washington Commanders | The Washington Commanders |
| 7920875 | `TOP5_TIE_CLUSTER` | 0.999 | SALESFORCE INC. | Salesforce Inc. |
| 7920877 | `TOP5_TIE_CLUSTER` | 0.999 | Salesforce - Primary | Salesforce - Primary |
| 7920972 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Saloon/Lodge Events 2026 | Yellowhouse Events |
| 7920974 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | SFOAO 2026 - Banquet Breakfast | SFOAO 2026 - FOOD TASTINGS |
| 7920985 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8017233938990089 | ATLWC 2026 CAPOLINEA BOOKINGS | ATLWC In House 2026 |
| 7920986 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8104537290260626 | ATLWC 2026 CAPOLINEA BOOKINGS | Avant Bookings 2026 |
| 7920988 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8161321494216699 | ATLWC 2026 IRD BOOKINGS | ATLWC In House 2026 |
| 7921092 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Shana Sperling Events | SSE |
| 7921235 | `LOW_CONFIDENCE_TOP1` | 0.53 | Purple Circle | Purple Orange PR |
| 7921255 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SANCC Restaurant Outlet Bookings | SANCC Restaurant Outlet Bookings |
| 7921257 | `LOW_CONFIDENCE_TOP1` | 0.53 | SANCC Restaurant Outlet Bookings | StorePoint Restaurant |
| 7921258 | `LOW_CONFIDENCE_TOP1` | 0.53 | Wynden Stark/GQR | Wynden Stark |
| 7921274 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Novo | Novo |
| 7921288 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | SWLA - Heel of the Boot | SHB |
| 7921301 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | ServiTech | Servite |
| 7921303 | `TOP5_TIE_CLUSTER` | 0.5997711601204783 | Sendflow | Perflow |
| 7921305 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SLB | SLB |
| 7921308 | `LOW_CONFIDENCE_TOP1` | 0.53 | SMX | SMX West |
| 7921332 | `LOW_CONFIDENCE_TOP1` | 0.53 | Wiz, Inc | Wiz, Inc |
| 7921341 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | PG Solutions | PG SOLUTIONS |
| 7921361 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Volvo Construction Equipment Haulers | Volvo Construction Equipment AB |
| 7921362 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7797811019639591 | Shandong Lingong Construction Machinery | Hitachi Construction Machinery |
| 7921379 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | WBD - TNTLA | WBD - TNTLA |

### Batch 3 (records 400..599)

- **In batch:** 200 | **Flagged:** 70 | **Flag rate:** 35.0%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7921421 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Nucleo Urbano Programa de Beneficios LTDA | Nucleo De Oncologia DA Bahia S / C Ltda |
| 7921422 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7977266696548375 | West Lake Financial | Westlake Financial |
| 7921423 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.760363858347038 | Wilshire Lane Capital | Wilshire State Bank |
| 7921429 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | SOS Financial Agency | SFA |
| 7921435 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Paceline Equity Partners | Partners Financial |
| 7921441 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Platform Accounting Group | PAG |
| 7921485 | `LOW_CONFIDENCE_TOP1` | 0.53 | WANY IRD | IRD |
| 7921488 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Yoshoku Q2 2026 | Yoshoku Events 2025 |
| 7921632 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.836751285273207 | Texas House of Democratic Caucus | Republican Liberty Caucus of Texas |
| 7921656 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8105778223554292 | Washington Society for Association Excellence | Washington State Association for Justice |
| 7921663 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Randstad Enterprise | Randstad Digital, LLC. |
| 7921688 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Society For Investigative Derma | SID |
| 7921689 | `TOP5_TIE_CLUSTER` | 0.999 | Society For Investigative Derma | Society For Investigative Derma |
| 7921690 | `TOP5_TIE_CLUSTER` | 0.7685551206804037 | PESTOLA 2026 - Affiliates | 2026 ACC Affiliates |
| 7921747 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.742177421184651 | Réseau québécois de recherche en soins palliatifs et de | Réseau des soins palliatifs du Québec |
| 7921770 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Schwartz and Company | Gregory J Schwartz & Company |
| 7921774 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | YA Group | YA Group |
| 7921797 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | RKO Sales Group | RSG |
| 7921799 | `LOW_CONFIDENCE_TOP1` | 0.53 | Solstice Advanced Materials Inc. | Special Materials Company |
| 7921807 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | PepperStone | Peppers |
| 7921826 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7401175695492301 | North Pacific Paper Corporation | Pacific Northern Environmental, LLC |
| 7921827 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Smurfit Westrock | Smurfit Westrock |
| 7921847 | `LOW_CONFIDENCE_TOP1` | 0.4478977003167071 | RxHealing | HealthX |
| 7922143 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Rawden Joint Ventures | Red Ventures Recruiting |
| 7922159 | `LOW_CONFIDENCE_TOP1` | 0.5295549323201414 | SNUG Incorporated | VNU, Incorporated |
| 7922160 | `TOP5_TIE_CLUSTER` | 0.999 | SLCCC Small Meeting | SLCCC Small Meeting |
| 7922163 | `LOW_CONFIDENCE_TOP1` | 0.53 | SNAMO - 2026 Exec Meeting | SNAMO 2023 - Exec Meetings |
| 7922167 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922173 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922197 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7452153454086996 | Women Leading Women Foundation | Womens Foundation |
| 7922272 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | Western Reserve Trust Company | Western Reserve Partner LLC |
| 7922275 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922276 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | VitalCare Infusion Services | VitalCare Infusion Services |
| 7922279 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Walton Ward Accommodations | Pastner Room Accommodations |
| 7922286 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | RWDI | RWD |
| 7922294 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | ATLWC 2026 Meetings | A2M |
| 7922367 | `LOW_CONFIDENCE_TOP1` | 0.53 | Workday, Inc. | Workday, Incorporated |
| 7922370 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Webisdom Group | Webisdom Group |
| 7922377 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7661262076119766 | Vital Edge Technologies | VitalEdge Technologies |
| 7922384 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Space42 | Space |
| 7922388 | `TOP5_TIE_CLUSTER` | 0.8497436655752315 | SPSA 2026 Affiliates | 2026 ACC Affiliates |
| 7922404 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Warrior Kido | Live Free Warrior |
| 7922424 | `LOW_CONFIDENCE_TOP1` | 0.53 | Samurai ATA Family Martial Arts | Alliance Martial Arts Center |
| 7922482 | `TOP5_TIE_CLUSTER` | 0.8995 | Viviana Viajes S.A. de C.V. | Viajes Lorimar S.A. de C.V. |
| 7922529 | `LOW_CONFIDENCE_TOP1` | 0.53 | When Brown Girls Lead | Four Brown Girls |
| 7922603 | `LOW_CONFIDENCE_TOP1` | 0.5396945976116757 | SHI Plastics Machinery de México, S.A. DE. C.V. | Metalsa Mexico |
| 7922614 | `LOW_CONFIDENCE_TOP1` | 0.5266657394678111 | QuickFi | KFI |
| 7922615 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Technical Maintenance Inc | TMI |
| 7922633 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | ROLLON Corp | Rollon Corporation |
| 7922642 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Raptor Wireline Services | RWS |
| 7922654 | `LOW_CONFIDENCE_TOP1` | 0.53 | VS&Co Board of Directors Summit | VS&Co Board of Directors Summit |
| 7922695 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pandora Y Flans | Pandora Y Flans |
| 7922726 | `LOW_CONFIDENCE_TOP1` | 0.53 | Solenis Limited | Solenis Front Line Leadership Training WA |
| 7922729 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Pièces d'auto Super | PDS |
| 7922751 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.876626886501177 | North Omaha Community Partnership | Greater Omaha Economic Development Partnership |
| 7922760 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Scouting America | National Football Scouting |
| 7922777 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8058678102402993 | Nubian Dive Club of Houston | Houston City Club |
| 7922782 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8541744112083918 | Ohio Association of Domestic Relations Judges | Ohio Common Pleas Judges Association |
| 7922828 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YMCA of the North | YMCA of the North |
| 7922847 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Site Travel LLC | STL |
| 7922856 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NINE21 Productions | Dance America Productions |
| 7922865 | `TOP5_TIE_CLUSTER` | 0.8057561942584268 | Nteractive | NTERACTIVE |
| 7922923 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | ","jornadas de expertos elaboracion de aparatos de yeso | ","jornadas de expertos elaboracion de aparatos de yeso |
| 7922972 | `LOW_CONFIDENCE_TOP1` | 0.53 | Sofwave | Firstwave |
| 7923048 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8424876784838292 | Horizon Meeting & Management Group | Meeting Management International |
| 7923063 | `LOW_CONFIDENCE_TOP1` | 0.5427550489558537 | Longtail Drives LLAC | My Driving 4 Life |
| 7923089 | `LOW_CONFIDENCE_TOP1` | 0.53 | Savara, Inc | Savara Inc |
| 7923092 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sheriff's Employees' Benefit Association (SEBA) | Sheriff's Employees' Benefit Association (SEBA) |
| 7923095 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Stateline Road | Road to California |
| 7923096 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Synergy Asset Management | SAM |

### Batch 4 (records 600..799)

- **In batch:** 200 | **Flagged:** 51 | **Flag rate:** 25.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7923098 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7927460564384949 | Texas Local Firefighters Foundation | Texas Community Charities Foundation |
| 7923101 | `LOW_CONFIDENCE_TOP1` | 0.5356266085909993 | The Shield Co | Clawson Shields Tours |
| 7923117 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TPC Management | TPC Transaction Processing Performance Council |
| 7923119 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tynan Group | Tynan Group |
| 7923129 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | WKS USA | W Standard USA |
| 7923130 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Xemax Surgical Products | XSP |
| 7923141 | `LOW_CONFIDENCE_TOP1` | 0.53 | On Behalf of PathGuide Technologies | Future Wireless Technologies |
| 7923196 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7724612318853403 | Misha Hawaii Home | Fair Housing Hawaii |
| 7923210 | `TOP5_TIE_CLUSTER` | 0.8 | Wedaways Travel | Wedaways Travel |
| 7923221 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8801378033272965 | Positive Physicians Insurance Company | Physicians Insurance Inc |
| 7923224 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sea Shell Pilates | SSP |
| 7923235 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7829794443023048 | Arizona Commission on Judicial Conduct | Arizona Commission on the Arts |
| 7923252 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | St. Christopher Catholic Parish | St. Patrick Catholic Community |
| 7923253 | `TOP5_TIE_CLUSTER` | 0.999 | TD SYNNEX Corporation | TD SYNNEX Corporation |
| 7923289 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.821866847078911 | Work Horse Land Development | Residential Land Development Practices |
| 7923344 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8518648200955959 | Orcas Capital Partners Limited | Alpha Capital Partners |
| 7923350 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7923351 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7923352 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7923356 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Typeform | Typeform |
| 7923363 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Youth Baseball Team / Westlake Village Gladiators | Westridge Ranchers Baseball Team |
| 7923366 | `TOP5_TIE_CLUSTER` | 0.999 | TAG - LA | TAG - LA |
| 7923368 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.519004447418134 | **BAND REQUEST** - Ying Yau B&C Party - Courtyard by Ma | Courtyard by Marriott Boston-Cambridge |
| 7923432 | `TOP5_TIE_CLUSTER` | 0.999 | The Westfield Group A | Westfield Group |
| 7923439 | `TOP5_TIE_CLUSTER` | 0.999 | UATP | UATP |
| 7923440 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Veterinary Innovative Partners | VIP |
| 7923445 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Westiminster School | Unquowa School |
| 7923492 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Institute For Strategic Lrning | ISL |
| 7923528 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | The White Dress Society | WDS |
| 7923545 | `LOW_CONFIDENCE_TOP1` | 0.5155365657865483 | Angus T. Brabham III | Midlands Honda |
| 7923606 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 108 Coaching Ltd | Focal Point Coaching |
| 7923628 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | HBA-Healthcare Businesswomen's Association | HBA Healthcare Businesswomens Association |
| 7923654 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.766332290642776 | Strategic Incentive Solutions | Incentive Enterprises |
| 7923657 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | TAM Advisory | ATX Advisory |
| 7923659 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | TRAVELIKO Uslugi Turystyczne | TUT |
| 7923664 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Vivid Sky Travel INC | VSTI |
| 7923671 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7581424593155854 | When Brown Girls Lead | Brown Girls Run |
| 7923672 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | World Boardgaming Championships | WBC |
| 7923678 | `LOW_CONFIDENCE_TOP1` | 0.53 | Ventur3 | Ventur3 (Formerly Plannerhero) |
| 7923688 | `LOW_CONFIDENCE_TOP1` | 0.5118275949262804 | BTCCHI0426 VGNRNBLW39M | WATCCHINC |
| 7923693 | `LOW_CONFIDENCE_TOP1` | 0.5007663072682664 | CHI21APR23APR26 L7NLLHBV8VZ | IND20APR23APR17 N4N64KVRY2H |
| 7923716 | `LOW_CONFIDENCE_TOP1` | 0.4808732728736451 | NKL WEXMAC RTOP 2862 H5NKXH55PJT | NKK SWITCHES |
| 7923733 | `LOW_CONFIDENCE_TOP1` | 0.53 | Wholesalecars.com | Wholesalecars.com |
| 7923751 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7604032271990324 | 2026 ITPEU Annual Investment Meeting LTNLRRRFJC8 | 2025 ITPEU Annual Benefit Fund Meeting |
| 7923753 | `LOW_CONFIDENCE_TOP1` | 0.53 | 510 wood Ave e | Wood Machinery Industry Association |
| 7923765 | `LOW_CONFIDENCE_TOP1` | 0.53 | BRG LLC | BRG LLC |
| 7923804 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | n3xt | NXTP |
| 7923823 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.810008487309818 | Spina Bifida Association of NYS | Spina Bifida |
| 7923824 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | SRO Motorsports America | SMA |
| 7923825 | `LOW_CONFIDENCE_TOP1` | 0.5485905410981485 | Stemline GCO F2F Apr2026 JLNPZ8NX54K | Stemline 2026 |
| 7923826 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sun River Healthcare | SRH |

### Batch 5 (records 800..999)

- **In batch:** 200 | **Flagged:** 74 | **Flag rate:** 37.0%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7923830 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Timothy Initiative | United Religious Initiative |
| 7923831 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tik Tok | Tik Tok |
| 7923832 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travel Event Staffing | Event Travel Management, N.A. |
| 7923833 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7826188377331608 | TRAVEL IDEA NEW YORK FIFA DVN62HQ6WCH | New York Travel Agent Fam Trip |
| 7923838 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7702122006352928 | Unilever 2026 Conference GKNTY4PNV4C | Unilever Conference Center |
| 7923839 | `WRONG_STATE_TOP1` | 0.7190971924706119 | Unitaid / World Health Organization | World Health Organization |
| 7923873 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7931481784986765 | Univeristy of Wisconsion-Madison | UNIV WI MADISON |
| 7923888 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Argo Group US | AGU |
| 7923967 | `LOW_CONFIDENCE_TOP1` | 0.5390753300638385 | Sundae Creative | Sunlife |
| 7923968 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | La Boite Rouge Vif | Bleu Blanc Rouge |
| 7923973 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Atlanta Costume Society | ACS |
| 7923982 | `LOW_CONFIDENCE_TOP1` | 0.53 | Executive Leadership Academy for Aspiring School Princi | Institute for Student Leadership |
| 7923986 | `LOW_CONFIDENCE_TOP1` | 0.5136581694618475 | First in Service, Dennie Miller Group , Portland, Mar 0 | Wilson Company |
| 7923987 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Friendship Missionary Baptist Church | FMBC |
| 7923999 | `LOW_CONFIDENCE_TOP1` | 0.5241855839938857 | AZ LAX - (University of Southern California) (Feb 14-15 | University Southern California |
| 7924004 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Ahmad and Susan | Susan Fong |
| 7924005 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7477252249938954 | ALG Vacations / Daniel Passariello Víctor Procencio / M | ALG VIAJES MEXICO / AMERICAN EXPRESS VACATIONS |
| 7924006 | `LOW_CONFIDENCE_TOP1` | 0.5351641834246772 | ALG Vacations / Naiane Kassidy Martija & Alexander Lee  | Jaho Vacations Mexico |
| 7924008 | `LOW_CONFIDENCE_TOP1` | 0.481117207705515 | ASSA - New Jersey Showcase Championship - Caldwell Univ | New Jersey City University Galleries |
| 7924009 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8270287920557985 | 104th Engineer Company - Vietnam Veterans | Vietnam Veterans of America Foundation |
| 7924016 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7545335737499856 | GRP valeverde Miami Julho/26 | GRP 2990 - MIAMI & EHTL 2025 |
| 7924029 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Abnormal AI March Block | AAMB |
| 7924032 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | SMRF - Individual | SMRF - Individual |
| 7924041 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8238458006375335 | Special Olympics Southern Califonia | Special Olympics S CA |
| 7924053 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Maccabi Games Tour | MGT |
| 7924056 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Unique Weddings and Events | UWE |
| 7924061 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9046805748325841 | Teenage Group - Galaxy Vacations | GALAXY VACATIONS INC. MEXICO |
| 7924066 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Smithfield Selma High School | Smithfield Selma High School |
| 7924067 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7580108935023526 | Small Group RFP for New Orleans | Code For New Orleans |
| 7924081 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sport Squad, Inc. | SSI |
| 7924087 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8456946579128746 | Tulane University Junior Night | Tulane University Alumni Relations |
| 7924088 | `TOP5_TIE_CLUSTER` | 0.6654991631705935 | T2C Sports - Wilmington Academy of Arts and Sciences (m | Academy of Arts & Sciences |
| 7924091 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7565017943086132 | USJN National Championships 2026 | HoopSource Presidents' Day National Championships 2025 |
| 7924095 | `HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.8 | Cassandra Brodeau | Cassandra Brodeau |
| 7924102 | `TOP5_TIE_CLUSTER` | 0.9040344149903817 | Berkshire Conference of Women Historians 2029 | Berkshire Conference of Women Historians |
| 7924103 | `LOW_CONFIDENCE_TOP1` | 0.5380525194186655 | Best in West Showcase / West Coast Showdown 2026 | Westport Country Playhouse |
| 7924104 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8566096974160039 | Big Country Quarter Horse Association | American Quarter Horse Association Affiliates |
| 7924105 | `LOW_CONFIDENCE_TOP1` | 0.5495091213187784 | BMG- Ossie Ware Mitchell MS | Digital Dealer |
| 7924106 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Board of Governors - CCCCO | BGC |
| 7924117 | `LOW_CONFIDENCE_TOP1` | 0.5238804421127907 | CISCA (Ceilings & Interior Systems Construction Associa | Valley Interior Systems |
| 7924122 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7589423320635942 | 1st Responder Conferences | CONFERENCES INC |
| 7924123 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Antioch Church | Antioch Community Church |
| 7924124 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7834226259818444 | ALG Vacations/2026  Amao70Getaway / Cancun | ALG Vacations |
| 7924125 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7698270240138887 | ICRI Delaware Valley Student Night | Delaware Valley Charter High School |
| 7924130 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | A.R.T. Global Education Ltd | CIES Comparative & International Education Society |
| 7924145 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | We Paint It | WPI |
| 7924149 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7482948855882186 | Voyage des leaders Energir | EM VOYAGE |
| 7924151 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7487929179189299 | Sharma, Smith & Gray, P.C. | YP Smith |
| 7924163 | `TOP5_TIE_CLUSTER` | 0.8995 | Toronto Request -Worldimension | Montreal Request-Worldimension |
| 7924174 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Sportograf Digital Solutions GMBH | Digital Art Solutions |
| 7924184 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8008718394191465 | UVET GBT/MSL/San Francisco | UVET GBT |
| 7924186 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7924187 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Smithfield-Selma High School | Smithfield-Selma High School |
| 7924188 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7924203 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | smti AG - Switzerland | SAS |
| 7924205 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8264571046997461 | SAT Mexico | SUMMIT FARMA ARMSTRONG Mayo 2026 | SAT Mexico | SnapX Incentive Tulum Mayo 2026 |
| 7924207 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | World Fitness Project | WFP |
| 7924212 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Synkwise | Synk |
| 7924225 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | Th | Terese Hoogoian |
| 7924235 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7924240 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Royalton Hotels and Resorts | Royal Resorts |
| 7924251 | `LOW_CONFIDENCE_TOP1` | 0.5085033447059459 | SMTI/World Cup26Group1/NEWYORK | links WorldGroup |
| 7924256 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8020392637195137 | Two Heads/RoomsOnly/New York | Two Heads |
| 7924295 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8184675010985125 | AD /  Group 2026 - Boston | THE AD CLUB BOSTON |
| 7924297 | `LOW_CONFIDENCE_TOP1` | 0.52 | Junta Nacional Lideres Estafeta 2026 marzo- Crearte | Junta Nacional |
| 7924301 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Affinity | AFFINITY TOURS |
| 7924303 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7538939670601859 | Alchemy - Convención Zurich Santander Nov 2026 | Zurich Santander |
| 7924304 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7503958403892431 | Access / ATS - Orlando | C / Life Orlando |
| 7924307 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Actigraph Holdings LLC | AHL |
| 7924309 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9070894049518334 | 2026 Academy Cup Tournament | FC PRIDE CUP TOURNAMENT |
| 7924310 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 ASCO Affiliates | 2026 ACC Affiliates |
| 7924363 | `LOW_CONFIDENCE_TOP1` | 0.5195195297160627 | BCD Spain/Tour Guitarrica de la Fuente/San Francisco | BCD Travel Spain Madrid |
| 7924404 | `LOW_CONFIDENCE_TOP1` | 0.53 | SuperFridge | SuperFridge |
| 7924406 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8059942293226238 | SWM/Pharma Investigator Meeting/Dallas - Chicago | SMS Chicago Investigator Meeting |

### Batch 6 (records 1000..1199)

- **In batch:** 200 | **Flagged:** 75 | **Flag rate:** 37.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7924409 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | SVdP Western Region | SWR |
| 7924413 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.741024916239643 | Match-accommodation | Miami South Beach | Jun 2026 | FIFA26 Accommodation Bureau Satellite Venue | JUN 2026 |
| 7924416 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.767135772705078 | Ventegra Client Conference | Ventegra |
| 7924421 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sea Shell Pilates | SSP |
| 7924427 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sedona Star Holistic Spa | Spa Nautica |
| 7924433 | `LOW_CONFIDENCE_TOP1` | 0.5307202104107905 | Smartsheet, Inc. | Smart Wool |
| 7924434 | `LOW_CONFIDENCE_TOP1` | 0.5026241982395164 | SMTI/World Cup26Group2/NEWYORK | links WorldGroup |
| 7924435 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Travora Global LLP | Travora Global LLP |
| 7924436 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TD Bank | TD Bank |
| 7924438 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7924439 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Thai Phi Nguyen | TPN |
| 7924440 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8286425155046215 | The Asian American Foundation TAAF - Leadership Summit | Conference on Asian Pacific American Leadership |
| 7924447 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7963264923711062 | UNCW Beach Volleyball at Georgia State | California State University, Long Beach Beach Volleybal |
| 7924451 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7816747124370977 | The Battery Network Board Meeting | Board of Directors Network |
| 7924456 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | IEA Interscholastic Equestrian Association | University of Findlay Equestrian |
| 7924467 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.768652548271916 | 247 TravelPro | Rio de Janeiro | jun 202 | PANAMORL - RIO DE JANEIRO - JUN  2026 |
| 7924468 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7601663589151091 | 2026 DNC Affiliates | 2023 AAD Affiliates |
| 7924472 | `LOW_CONFIDENCE_TOP1` | 0.53 | TSO | TSO |
| 7924485 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | National Eagle Scout Association | NESA |
| 7924486 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7463075392719978 | University of Arkansas Forth Smith Upward Bound Program | University of Arkansas Fort Smith |
| 7924487 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Streamline Hospital Services | SHS |
| 7924493 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.77006477563925 | All Rise - United States | United States Congressman |
| 7924494 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Americain Society of Questioned Document Examiners | Southwestern Association of Forensic Document Examiners |
| 7924507 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Beyond Health Parnters | BHP |
| 7924509 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Christ Centered Chiropratic | CCC |
| 7924511 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9131615812976928 | CION Investment Corporation | Cion Investments |
| 7924530 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7975499329536002 | AMEX M&E Mexico/Summit Hematología 2026 | AMEX M&E Mexico/OFF SITE 2026 |
| 7924531 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | All Star Life Group | All Star Group |
| 7924532 | `LOW_CONFIDENCE_TOP1` | 0.5222941940548045 | Anaplan | ANA |
| 7924540 | `LOW_CONFIDENCE_TOP1` | 0.4816988604432798 | BCD / JJ - ID 186769 - JHNKN4W9QFV - Treinamento de Pal | Paley Honors 2025 |
| 7924541 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8138927122479309 | Bankers - Life Workshop Cleveland | Bankers Life and Casualty Annual |
| 7924542 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8188679188367995 | Barbados Agricultural and Marketing Development Corpora | Barbados Agricultural Society |
| 7924544 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | BRAINBox Solutions, Inc | BSI |
| 7924546 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7674511648100605 | Brook green - Missao Internacional - San Francisco | David Green Organization San Francisco |
| 7924547 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.793385195001336 | Av Business | Convención Midea 2026 | AV Business -  grupo Argentina-año 2026 |
| 7924548 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Bryant Park Event Room Block | Tinley Park Wresting Room Block |
| 7924549 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7480231313685226 | Carenzi Vanues/CDM Media new enquiry/New York | WNET New York Public Media |
| 7924552 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7470443427532266 | Cabo Offsite (5ECR9Q) | A & O Offsite |
| 7924562 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7849815063256775 | Southern Regional Council of Carpenters | Central South Regional Council |
| 7924577 | `LOW_CONFIDENCE_TOP1` | 0.5379183408030463 | The Association of Technology, Management (ATMAE) | ATMIA |
| 7924593 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7468751123973302 | ALG Vacations / Saasha Day / Cancun | ALG Vacations |
| 7924598 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7627529609990188 | Grupo NYC – Parejas – Agosto 2026 | FIFA World Cup 2026 Group – NYC |
| 7924606 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5072884343964856 | First in Service, Dennie Miller Group , Atlanta, Feb 08 | US Census Bureau Atlanta Regional Service Center |
| 7924608 | `LOW_CONFIDENCE_TOP1` | 0.5378497973154496 | EMI / Group Request / New Orleans, LA | EMI Global USA |
| 7924614 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | TransDigm Group, Inc. | TGI |
| 7924623 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8049487280588433 | SANCC Restaurant Outlet Bookings | Harth Lounge & Restaurant Bookings |
| 7924625 | `LOW_CONFIDENCE_TOP1` | 0.53 | Starpower, LLC | RE Star Power, LLC |
| 7924630 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8139653305437435 | Georgia Coalition of Black Chambers | Georgia Black Republican Council |
| 7924642 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8017624536711033 | Andrea Dodson Acct Management | Andrea Dodson Inhouse Account |
| 7924646 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Blue Sky Travel | BST |
| 7924650 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travelicious Tours Pvt. Limited | Travelicious Tours PVT Limited |
| 7924652 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Solventum | Solventum |
| 7924695 | `LOW_CONFIDENCE_TOP1` | 0.5448634515456241 | TocinoyAguacate | Turismo Miya |
| 7924699 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Apex Agency | APEX Agency |
| 7924713 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | UKG Inc. | UKG Inc. |
| 7924720 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Central States Manufacturing | CSM |
| 7924721 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | WSP | WLU Student Publications |
| 7924724 | `LOW_CONFIDENCE_TOP1` | 0.5446503006975747 | Subtotal | WASHTO |
| 7924728 | `LOW_CONFIDENCE_TOP1` | 0.5342183783129115 | Tenaris EMPRESA DEL GRUPO TECHINT | Techint Argentina |
| 7924729 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Sauer Group, LLC | SGL |
| 7924746 | `LOW_CONFIDENCE_TOP1` | 0.53 | HSMWINGPAC - Helicopter Maritime Strike Wing, Pacific | HSMWINGPAC - Helicopter Maritime Strike Wing, Pacific |
| 7924749 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8610895747990464 | USMC 6th Marine Corps Recruiting District | United States Marine Corps Eastern Recruiting |
| 7924763 | `LOW_CONFIDENCE_TOP1` | 0.53 | NewEdge Wealth | Newly Wealthy |
| 7924770 | `TOP5_TIE_CLUSTER` | 0.9194000000000001 | Banorte Wealth Management | CBG Wealth Management |
| 7924799 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Waypoint, LLC | CrossPointe, LLC |
| 7924809 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.760826438809838 | Asociación Nacional de Empleados del Banco de la Repúbl | Asociación de bancos públicos y privados de la repúblic |
| 7924818 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5106486607719205 | Upace | NPACE |
| 7924824 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8268331975113862 | Connecticut Society of Health System Pharmacists | VA Connecticut Health Care System |
| 7924825 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | BIOS Life, Inc. | BLI |
| 7924831 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Society of Epidemiologic Research | SER |
| 7924839 | `LOW_CONFIDENCE_TOP1` | 0.531996033911107 | AgentSync | See Agency and Agent |
| 7924840 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7877939759171235 | Développement informatique Clic Assure Inc. | Clic Assure |
| 7924841 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | National Association Goosehead Agents | NAGA |
| 7924852 | `LOW_CONFIDENCE_TOP1` | 0.53 | Boydorr Nutrition | Mead Johnson Nutrition Colombia |
| 7924858 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Adept Fasteners | Adept Technologies |

### Batch 7 (records 1200..1399)

- **In batch:** 200 | **Flagged:** 62 | **Flag rate:** 31.0%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7924860 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.766570188725529 | World's Poultry Foundation | International Poultry Council |
| 7924873 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7865830009633845 | Séptimo Cielo CWP SAS | El Cielo SAS |
| 7924915 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7924921 | `LOW_CONFIDENCE_TOP1` | 0.53 | Cardata | Cardata |
| 7924942 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Ahava International | Voyage International |
| 7924957 | `TOP5_TIE_CLUSTER` | 0.7955893337726593 | Stribe Dental | Dental Hygiene |
| 7924965 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Sol Dance Collective | SDC |
| 7924967 | `LOW_CONFIDENCE_TOP1` | 0.53 | Brittney Beckman | Honer-Beckman Wedding |
| 7924973 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Air Europa Cargo | AEC |
| 7924976 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bonita Trip | Bonita Travels |
| 7924981 | `LOW_CONFIDENCE_TOP1` | 0.47150621051387853 | Turitika | Artika |
| 7924992 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | World Tour Logistics LLC | Impact Logistics |
| 7924998 | `LOW_CONFIDENCE_TOP1` | 0.53 | UGS3 | UGS3 Group |
| 7925012 | `LOW_CONFIDENCE_TOP1` | 0.53 | Altec, Inc. | Altec |
| 7925020 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Thermomix | Thermomix |
| 7925023 | `LOW_CONFIDENCE_TOP1` | 0.53 | Technoprofil | Technorm |
| 7925030 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Working Minds | Construction Working Minds |
| 7925040 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Brook Health | Brook Health |
| 7925043 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Azazie Bridal ULC | ABU |
| 7925045 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Tecky Brains ONG | TBO |
| 7925048 | `TOP5_TIE_CLUSTER` | 0.999 | Alliant Healthcare Solutions | Alliant Healthcare Solutions |
| 7925049 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Chinese Business Women's Association | CBWA |
| 7925053 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | United Suicide Survivors International | USSI |
| 7925054 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skills / Compétences Canada | Skills / Compétences Canada |
| 7925065 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Appui de pointe | ADP |
| 7925073 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7562767736144237 | Unlock your Power within | Step Into Your Power |
| 7925095 | `LOW_CONFIDENCE_TOP1` | 0.53 | Venomere Group | Venomere Group |
| 7925099 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Sourcing on behalf of One Resource Group (An Integrity  | ONE RESOURCE GROUP |
| 7925105 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.75 | We Plan It | WPI |
| 7925113 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | Woodbrey Family Travel | Family and Friends Travel |
| 7925118 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Shionogi Inc | Shionogi Inc |
| 7925299 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | General HVAC Solutions America, Inc. | General HVAC Solutions America, Inc. |
| 7925301 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | Harbor Group International | Pearl Harbor |
| 7925317 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.807518572561232 | Bank Art Fair Organizing Committee | Small Business Fair Organizing Committee |
| 7925337 | `TOP5_TIE_CLUSTER` | 0.999 | PSI CRO | PSI CRO |
| 7925352 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Waters Corporation | Waters Corporation |
| 7925353 | `LOW_CONFIDENCE_TOP1` | 0.53 | Waters Corporation | Water Corporation - USA |
| 7925380 | `TOP5_TIE_CLUSTER` | 0.999 | Accuity | Accuity |
| 7925387 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7642208118394676 | Board Leadership for Destinations LLC | Destination Leadership Consortium |
| 7925391 | `LOW_CONFIDENCE_TOP1` | 0.53 | Cumming Group | Cummins |
| 7925393 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Grindr | Grindr |
| 7925394 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | INSTITUTE FOR HEALTHCARE ADVANCEMENT | IHA |
| 7925395 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Internatonal Society of Caricature Artists | ISCA |
| 7925396 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Me Plus Ultra | Material Plus |
| 7925414 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Travere Therapeutics, Inc. | TTI |
| 7925433 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | 114446 HPN - May 2026 Meeting | HPN Board Meeting May 2021 |
| 7925434 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7838061982128464 | AERC National Conference | AERC |
| 7925435 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7964410167249059 | Alzheimer's Soceity of Santa Clara | SANTA CLARA ALUMNI ASSOCIATION |
| 7925436 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Beverly Homes Gratitude | BHG |
| 7925437 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | CATHEDRAL OF FAITH FAMILY PRAISE CENTER INTERNATIONAL | International Cathedral of Faith Fellowship |
| 7925440 | `TOP5_TIE_CLUSTER` | 0.999 | Good Feet Store | The Good Feet Store |
| 7925442 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Kingston MYRES - 2024 | MYRES Wedding 2022 |
| 7925491 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | New Greater Deliverance Church | House of Deliverance Church |
| 7925492 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | New Greater Deliverance Church | Next Level Church |
| 7925493 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | New Source Network | NSN |
| 7925495 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | NPS Panels | NPS Air Resources Division |
| 7925496 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Prospect Community Church | PCC |
| 7925500 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Top Gun Sports | TGS |
| 7925505 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Winn Companies LLC | WCL |
| 7925506 | `LOW_CONFIDENCE_TOP1` | 0.53 | Asa Reign Event Management | Special Events Management |
| 7925509 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 North American Sales Meeting | NA Sales Meeting Jan 2022 |
| 7925513 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | America’s Auto Auction | America's Auto Auction |

### Batch 8 (records 1400..1599)

- **In batch:** 200 | **Flagged:** 55 | **Flag rate:** 27.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7925526 | `LOW_CONFIDENCE_TOP1` | 0.53 | BMT CTN | CTN Travels |
| 7925527 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Bond Events Corporation | BEC |
| 7925540 | `LOW_CONFIDENCE_TOP1` | 0.5289015787688747 | CHCollective | CHF |
| 7925549 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Constellatin Brands | Best Life Brands |
| 7925553 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Cruise One/Dream Vacations | COV |
| 7925556 | `LOW_CONFIDENCE_TOP1` | 0.5339974973535191 | Deep Trekker | Discovery Travel |
| 7925558 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Destination Possible Travel | DPT |
| 7925559 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Diabetes Research Connection | DRC |
| 7925561 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8066758113056421 | DiSanto Priest & Co/Bentley Wealth Advisors | Bentley Wealth Advisors LLC |
| 7925565 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | EcoShield Pest Control | EPC |
| 7925566 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Essence Consulting, Inc. | ECI |
| 7925567 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | F.L.Putnam Investment Management Company | Investment Management Institute |
| 7925569 | `LOW_CONFIDENCE_TOP1` | 0.53 | Federato | Federato Technologies |
| 7925576 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Fullbay | Fullbay |
| 7925581 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Hanna Interpreting Services | HIS |
| 7925582 | `HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.79702707529068 | Hometown Heroes Inc | Hometown Heroes |
| 7925583 | `WRONG_STATE_TOP1, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.8101956230991973 | Hotel Engine | Hotel Engine |
| 7925587 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.4433157870138253 | ICE-T (Immune Cell Effector Therapy) Conference | T Cell Lymphoma Forum |
| 7925588 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Illume Events | On Points Events |
| 7925595 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Katie Brown Educational Program | EDUCATIONAL PROGRAMS |
| 7925604 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Lunamare Escapes | PORTER ESCAPES INC. |
| 7925605 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Luxe Events by Crystal | LEC |
| 7925607 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Main Street Automotive | MSA |
| 7925608 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Margaritaville Hotel Nashville | MHN |
| 7925610 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | McCarthy Holdings | McCarthy Holdings, Inc. |
| 7925615 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | naturaLED | Natural High |
| 7925616 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Neoss | NEOS |
| 7925620 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NorthStar Insurance Services, Incorporated | NorthStar Insurance Services, Incorporated |
| 7925622 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | NWN | Naturals Who Network |
| 7925629 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | OGT | ogt |
| 7925642 | `LOW_CONFIDENCE_TOP1` | 0.52 | Osaic 821903 Osaic 2026 Ovation Incentive Apr2026 VDN9H | 821903 Osaic 2026 Ovation Incentive |
| 7925643 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8469271827340127 | Pacific College of Health and Science | Pacific College |
| 7925652 | `LOW_CONFIDENCE_TOP1` | 0.53 | Pathward | Pathward, N.A. |
| 7925659 | `LOW_CONFIDENCE_TOP1` | 0.53 | Pellera Technologies | BBN TECHNOLOGIES |
| 7925675 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | See Agency and Agent | SAA |
| 7925676 | `LOW_CONFIDENCE_TOP1` | 0.5364392540242862 | Segway Navimow | Savi |
| 7925677 | `LOW_CONFIDENCE_TOP1` | 0.53 | Selector AI | Impact Selector |
| 7925679 | `LOW_CONFIDENCE_TOP1` | 0.53 | SESI Schools | Uncommon Schools |
| 7925680 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Shasta Dental Services | SDS |
| 7925694 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sojourn Adventures | Discover   Adventures |
| 7925695 | `LOW_CONFIDENCE_TOP1` | 0.544422353180443 | Southern Sleep Society 2016 | Southern Home Services |
| 7925697 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spellers Freedom Foundation | Freedom Writers Foundation |
| 7925720 | `LOW_CONFIDENCE_TOP1` | 0.5244174167237294 | The Gilded Birds | Eagle |
| 7925725 | `LOW_CONFIDENCE_TOP1` | 0.53 | Tonies US | A Better Us |
| 7925726 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Towie Travel | Great Adventure Travel |
| 7925728 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Travel Journey KC | TJK |
| 7925729 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.816863770969599 | University of California at San Diego All Campus | University of California San Diego Medical Center Hills |
| 7925760 | `LOW_CONFIDENCE_TOP1` | 0.53 | Xylem, Inc. | Xylem, Incorporated |
| 7925768 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Institute of Scrap Recycling Industries d/b/a Recycled  | Institute of Scrap Recycling Industries, Gulf Coas |
| 7925775 | `LOW_CONFIDENCE_TOP1` | 0.53 | Saint Vincent de Paul School | Saint Anthony School |
| 7925776 | `LOW_CONFIDENCE_TOP1` | 0.53 | Select Service | Select Hospitality |
| 7925798 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | McLamore-Clark Margaret | MCM |
| 7925804 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Henry Linda | Linda Henry |
| 7925811 | `MATCH_TYPE_LOW_CONF` | 0.79 | MUB | MUFG - Union Bank |
| 7925820 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.911595178822574 | Alarm.com Security Forum | Information Security Forum |

### Batch 9 (records 1600..1799)

- **In batch:** 200 | **Flagged:** 47 | **Flag rate:** 23.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7925823 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.43964120130702045 | TSA via CLC/Corpay - Snow storm support - DULLES | Snow Storm Technologies |
| 7925834 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Online Bookings - WASSL | OBW |
| 7925835 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Online Bookings - WASSL | OBW |
| 7925836 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Online Bookings - WASSL | OBW |
| 7925837 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Online Bookings - WASSL | OBW |
| 7925838 | `TOP5_TIE_CLUSTER` | 0.8796000000000002 | Online Bookings - WASSL | Online Bookings - RLSC |
| 7925839 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Online Bookings - WASSL | OBW |
| 7925842 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | NetImpact Strategies, Inc. | NSI |
| 7925844 | `LOW_CONFIDENCE_TOP1` | 0.53 | Mid-South Gifted Academy | Academy Travel |
| 7925845 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Fonix Travel Agency | FTA |
| 7925863 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | JOICY NAILS SPA | Shine Massage and Spa |
| 7925868 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Marie Moran & Company, LLC | Marie Moran & Company, LLC |
| 7925869 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7781557041988743 | Integrated Professional Solutions | Integrated Business Systems |
| 7925872 | `LOW_CONFIDENCE_TOP1` | 0.53 | 14U Junior Dawgs | Diamond Dawgs Baseball Team |
| 7925876 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Morven Park | Morven Park |
| 7925877 | `LOW_CONFIDENCE_TOP1` | 0.53 | Tough2gether Foundation | Impact Foundation |
| 7925883 | `LOW_CONFIDENCE_TOP1` | 0.5330511248421699 | KabaFusion | Perfusion International |
| 7925887 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | The Apostolic Church International | ACI |
| 7925896 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Trump National Golf Club Washington DC LLC | Trump National Golf Club Washington DC LLC |
| 7925897 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Nothing Bundt Cakes | NBC |
| 7925900 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | T.White Parker | T.White Parker |
| 7925902 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7426718200364935 | OHCAL Mid-Atlantic | Pan Atlantic |
| 7925903 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | OHCAL Mid-Atlantic | OMA |
| 7925904 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | OHCAL Mid-Atlantic | OMA |
| 7925919 | `TOP5_TIE_CLUSTER` | 0.999 | SFK Tours | SFK Tours |
| 7925922 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.801834980930175 | Prestige MJM FAM 2026 | Prestige Travel FAM Trip |
| 7925931 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8128481489320596 | United Suicide Survivors International | Survivors of Suicide Loss |
| 7925935 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.805617158996788 | COST Spring 2027 Meeting | Spring Meeting 2022 |
| 7925947 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | 2029 Marketing Sales & Service Summit & | Contact Discovery Sales & Marketing Summit |
| 7925948 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | AAID Feb Meeting | AFM |
| 7925950 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | AMA 2028 Annual Conference | AMA Executive Conference Center |
| 7925951 | `TOP5_TIE_CLUSTER` | 0.8995 | 2026 Worldwide Sales Kick Off Meeting | 2023 Genie Sales Kick-Off Meeting |
| 7925956 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7875929367898938 | 2026 PRO-Talk Live | 2026 Plain Talk Affiliates |
| 7925958 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.835536228471539 | BMO FINANCIAL GROUP GLOBAL MASTER | BMO Global Asset Management |
| 7925960 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | CompRe Group | Compre Group |
| 7925962 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Howden Re | Howden Re |
| 7925965 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Financial Services Incentive Program | 2003 Qwest Incentive Program |
| 7925968 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8046257704822032 | Mayo Clinic 2026 Genetics in Cardiology | Mayo Clinic 2026 Critical Care Cardiolog |
| 7925972 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | National CineMedia - NCM | NCN |
| 7926063 | `LOW_CONFIDENCE_TOP1` | 0.5377324399253417 | AMAYA PRIVATE DINING 2026 | Bahia Guests 2023 |
| 7926064 | `LOW_CONFIDENCE_TOP1` | 0.53 | AMAYA PRIVATE DINING 2026 | Signia Private Dining |
| 7926071 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Jabra Ghanayem Engagement Party RB | Vora Engagement Party |
| 7926080 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Lucky Sun Event | LSE |
| 7926089 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9070894049518334 | Morgan Stanley Breakfast Meeting | Morgan Stanley Dean Water |
| 7926093 | `LOW_CONFIDENCE_TOP1` | 0.52 | Maya Gas Group (B Party) | Maya Gas Group (B Party) |
| 7926107 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7787604705977602 | Randolph Street Market Festival | 130 East Randolph Street |
| 7926108 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Recruitment Coach | Beyond Being Coach |

### Batch 10 (records 1800..1999)

- **In batch:** 200 | **Flagged:** 48 | **Flag rate:** 24.0%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7926120 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7967352428029533 | The Kayla Carter Group | Regina Carter |
| 7926136 | `HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.76 | Super Bowl Viewing | Super Bowl Viewing Party |
| 7926139 | `LOW_CONFIDENCE_TOP1` | 0.5388128606439659 | Tejash Patel's Diwali Party | Prachi Patel and Neel Patel's wedding |
| 7926158 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8312218671726868 | Work Crew 2026 HP8441484 | Anissa P Work Crew HP8288089 |
| 7926171 | `TOP5_TIE_CLUSTER` | 0.8995 | Terns Ad Board | March Ad Board |
| 7926174 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7776832351317773 | Talent Agency UTA Executive Retreat | Talent Agency |
| 7926175 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7798142799964318 | Taylormade Golf Winter 2026 NSM | TaylorMade Golf |
| 7926178 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.758746162789819 | Sophie and Magma Room Block | Karina Luna Room Block |
| 7926179 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7934125227405334 | Sound Physicians 2026 Medical Director S | SOUND PHYSICIANS |
| 7926181 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8313841327845416 | Skylight Marketing Retreat 2026 | Beacon Digital Marketing Retreat |
| 7926183 | `LOW_CONFIDENCE_TOP1` | 0.53 | Scorpion | Scorpion |
| 7926191 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8176048733856636 | TW Metal Summit 2027 | TW Metal Global |
| 7926201 | `TOP5_TIE_CLUSTER` | 0.999 | VF Corp | VF CORP |
| 7926205 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Waste Connections Lonestar, Inc. | WCLI |
| 7926229 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7680864125251023 | Procore Culture Academy 2026 | AZ Culture Academy |
| 7926233 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7805109692739547 | Radisson Blu Corporate Group Overflow | Radisson Blu |
| 7926234 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Raffles and Fairmont | Raffles and Fairmont |
| 7926235 | `TOP5_TIE_CLUSTER` | 0.999 | Paymentus | Paymentus |
| 7926236 | `LOW_CONFIDENCE_TOP1` | 0.53 | PCX Austin | Austin Travel |
| 7926239 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8115027481195602 | Pfizer Leaders Meeting 2026 | Pfizer Meeting 2025 |
| 7926240 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Pharma Party Chi | PPC |
| 7926242 | `TOP5_TIE_CLUSTER` | 0.6159726865780956 | PharmOne | Pharma |
| 7926249 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7842841577536688 | Nutrien Ag Solutions Learning Conference | Nutrien US LLC / Nutrien Ag Solutions |
| 7926259 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Netwrix | Netwrix |
| 7926261 | `TOP5_TIE_CLUSTER` | 0.999 | Newport Group | Newport Group |
| 7926264 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8110833102426495 | Nourish Company-wide All-Hands Meeting 2 | Janssen Business Development All Hands Meeting |
| 7926267 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Meanwhile | Meanwhile Incorporated |
| 7926269 | `TOP5_TIE_CLUSTER` | 0.8995 | Mercury Marketing Team Offsite | Replicated Marketing Team Offsite |
| 7926271 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Management Off-site | Management Committee Off-site and Teambuilding |
| 7926272 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8319072950509462 | Manager Meeting Dec 2026 | Manager Meeting 2024 |
| 7926285 | `TOP5_TIE_CLUSTER` | 0.999 | National Alliance of Wound Care & Ostomy | National Alliance of Wound Care & Ostomy |
| 7926295 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8429888293941745 | Luis Soto Group | Team Soto |
| 7926297 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | KENX | KENX |
| 7926307 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7813102522721658 | Adidas Golf Summer 2026 NSM | Adidas Golf |
| 7926325 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7750255948548963 | 11th Annual Diabetes and Obesity Conf | Obesity Conf |
| 7926332 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | 8am | 8 A marketing |
| 7926333 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.756633683605623 | 2029 NYLI Partners Meeting | Global Meeting Partners |
| 7926335 | `LOW_CONFIDENCE_TOP1` | 0.5134188182788642 | 46 room nights HE-163281 | Room 1520 |
| 7926340 | `TOP5_TIE_CLUSTER` | 0.8995 | Banner Partner Retreat | 2024 Partner Retreat |
| 7926341 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7569955245905877 | Barnes Beverage Group | American Beverage Association |
| 7926348 | `TOP5_TIE_CLUSTER` | 0.999 | Baker Tilly Advisory Group, LP | Baker Tilly Advisory Group, LP |
| 7926351 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Annual Conference 2026 | 2026 COSA Annual Conference |
| 7926352 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8109719889014959 | Annual National Hospitalist Confere | Hospitalist Conf |
| 7926354 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Annual Women’s Health Conference | U.S. Women’s Health Alliance |
| 7926356 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7905037050897425 | APO Associate Partner Orientation | Corporate Partner Orientation |
| 7926357 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8129612102052136 | April 2026 OVERFLOW 114586 | 2026 Converge - Overflow Block |
| 7926358 | `TOP5_TIE_CLUSTER` | 0.999 | Ariat International | Ariat International |
| 7926366 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9630506660549188 | Bermuda Realty Company Limited | Coldwell Banker Bermuda Realty |

### Batch 11 (records 2000..2199)

- **In batch:** 200 | **Flagged:** 65 | **Flag rate:** 32.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7926368 | `TOP5_TIE_CLUSTER` | 0.8995 | Board Meetings ID1241165 | Advisory Board Meetings |
| 7926370 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8173023035526276 | C&C Club Annual Outing | C Club |
| 7926375 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.792805106142782 | Chase Travel Group ELT Meeting | Chase Travel |
| 7926376 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8264409296092663 | Business Meeting 2026 HP8425575 | HP Business Meeting |
| 7926377 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8258172556169519 | Business Meeting 2026 HP8448246 | Business Meeting September 2025 |
| 7926398 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | IEEE Systems Council | ISC |
| 7926408 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8053023713301166 | Hims Inc. Quality Offsite 2026 | 2026 Market Access Offsite |
| 7926412 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8475606555392345 | HomeVestors 2026 Chicago Summit | Next Level Chicago 2026 |
| 7926422 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8008625973601525 | Geotab Grow January 2027 | Geotab January Meeting |
| 7926424 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8092259976235471 | Fish & Richardson Office Managers | Fish and Richardson Law Firm |
| 7926426 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Ford Regional Golf Event | Beau Townsend Ford Golf Group |
| 7926441 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8244173113129619 | FERMÀT Client Advisory Board Retreat | Risk Client Advisory Board Meeting |
| 7926445 | `LOW_CONFIDENCE_TOP1` | 0.53 | DeepLearning AI | Replicant AI |
| 7926446 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.779232464157618 | Dealer Tire Audi Incentive 2026 | Dealer Tire |
| 7926447 | `LOW_CONFIDENCE_TOP1` | 0.5492428142932325 | Cyera | CYTS |
| 7926453 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Customer Advisory Board (CAB) 2026 | Evotek 2025 Customer Advisory Board |
| 7926461 | `LOW_CONFIDENCE_TOP1` | 0.5373767822772393 | DrafKings | Clean Lab |
| 7926462 | `LOW_CONFIDENCE_TOP1` | 0.5373767822772393 | DrafKings | Clean Lab |
| 7926477 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.807384562040667 | Cornerstone Research Officer Meeting | Cornerstone Research Office Retreat |
| 7926479 | `LOW_CONFIDENCE_TOP1` | 0.53 | Corporate Event for MB 1N | Corporate Event Strategies |
| 7926483 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.753587180429028 | Complete Management Strategies, LLC | Strong Strategies Consulting LLC |
| 7926485 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Affiliated Independent Distributors, Inc | Affiliated Independent Distributors, Inc |
| 7926487 | `TOP5_TIE_CLUSTER` | 0.999 | Bayer Meeting Management Team @ Maritz T | Bayer Meeting Management Team @ Maritz T |
| 7926488 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Bayer Meeting Management Team @ Maritz T | Bayer Meeting Management Team @ Maritz T |
| 7926492 | `LOW_CONFIDENCE_TOP1` | 0.53 | Cathy Palmateer | Cathy Palmateer |
| 7926500 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | GTC | GCF Training Company |
| 7926502 | `LOW_CONFIDENCE_TOP1` | 0.53 | Ideal Living | Assisted Living |
| 7926507 | `LOW_CONFIDENCE_TOP1` | 0.5424920915310182 | J Shay Event Solutions | J.Shay Events |
| 7926510 | `LOW_CONFIDENCE_TOP1` | 0.53 | MAMS | MAMS |
| 7926514 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skirt | Skirt |
| 7926515 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Meeting Advocate | Advocate |
| 7926516 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Optimus Travel | Destiny Travel |
| 7926517 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travel by JB | JB Travel |
| 7926518 | `LOW_CONFIDENCE_TOP1` | 0.53 | Arcoro | Arcoro |
| 7926519 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | CoC Strategy Meeting | CSM |
| 7926524 | `LOW_CONFIDENCE_TOP1` | 0.5421669602394105 | DALPC* | DALPC |
| 7926530 | `LOW_CONFIDENCE_TOP1` | 0.53 | DALPC* | DALPC |
| 7926531 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | DALPC* | DALPC |
| 7926539 | `TOP5_TIE_CLUSTER` | 0.999 | Groups360 | Groups360 |
| 7926546 | `TOP5_TIE_CLUSTER` | 0.999 | Progress Residential | Progress Residential |
| 7926550 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 ND Golf Trip | Total Golf Travel |
| 7926551 | `LOW_CONFIDENCE_TOP1` | 0.53 | 2026 Q1 Sales and Events | SANQQ SALES 2023 |
| 7926555 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Avalon Bay Alumni | ABA |
| 7926556 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Baker & Karvala Ceremony | Slivka / Baker Wedding |
| 7926557 | `LOW_CONFIDENCE_TOP1` | 0.53 | Bishops Bay Country Club | Green Bay Country Club |
| 7926560 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Breakfast Ball Invitational | BBI |
| 7926564 | `LOW_CONFIDENCE_TOP1` | 0.5420659666964627 | County and Tribal Veterans Service Offices of Wisconsin | Old Wisconsin |
| 7926565 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Databricks Inc. | Databricks Inc. |
| 7926568 | `LOW_CONFIDENCE_TOP1` | 0.5468291907006042 | Eaker Group | EAI Corporation |
| 7926569 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | EPIC 21 | Century 21 Moves |
| 7926570 | `LOW_CONFIDENCE_TOP1` | 0.52 | Epic Girls Trip with Lesley | Girls Trip 2025 |
| 7926574 | `LOW_CONFIDENCE_TOP1` | 0.53 | Hooper Corporation | Hooper Corporation |
| 7926576 | `LOW_CONFIDENCE_TOP1` | 0.53 | Infinity Natural Resources | Infinity Natural Resources |
| 7926577 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | James Wenk Outing | Pelkey Golf Outing |
| 7926578 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Karen Keyes Group | KKG |
| 7926584 | `LOW_CONFIDENCE_TOP1` | 0.53 | Maguigan Group | MAG Group |
| 7926587 | `LOW_CONFIDENCE_TOP1` | 0.53 | Mike Jordan Outing | Griffin Outing |
| 7926588 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Mississippi Boys GOLF | MBG |
| 7926589 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5178819858110869 | MojoPMM | Mojoup |
| 7926593 | `LOW_CONFIDENCE_TOP1` | 0.53 | Oppidan | Oppidan |
| 7926598 | `LOW_CONFIDENCE_TOP1` | 0.53 | Ravenna Group | Ravenna Group |
| 7926602 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | RVR Competition, LLC | RCL |
| 7926606 | `LOW_CONFIDENCE_TOP1` | 0.53 | The Kohler 8 | Kohler Company - HQ |
| 7926608 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Third Cost Wealth Advisors | Independent Financial Advisors |
| 7926609 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Thrivent | Thrivent |

### Batch 12 (records 2200..2399)

- **In batch:** 200 | **Flagged:** 51 | **Flag rate:** 25.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7926610 | `LOW_CONFIDENCE_TOP1` | 0.53 | TO Far and Sure Golf Tours | Golf Tour |
| 7926611 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | USTA | UNITED STATES TRAVEL ASSOCIATION |
| 7926613 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vivian Calusinski Yoga | Yoga Six |
| 7926637 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.764168739160042 | 1st Century Bank | First National Bank |
| 7926641 | `WRONG_STATE_TOP1, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.8057561942584268 | Accuity LLP | Accuity LLP |
| 7926644 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.784133551120758 | ACTA | American College Theatre Association |
| 7926645 | `TOP5_TIE_CLUSTER` | 0.999 | Agility Fuel Solutions | Agility Fuel Solutions |
| 7926647 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Allocate | Allocate |
| 7926650 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | American Missionary Church | AMC |
| 7926659 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8199365046933764 | Bellflower Unified School District | Bellflower High School |
| 7926666 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Capital Alliance Mastermind | CAM |
| 7926669 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | CMX | CMS |
| 7926670 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.758512936776362 | Colorado Surgical Institute | Colorado Clinic |
| 7926676 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Convergint Technologies LLC | CTL |
| 7926677 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Cornerstone Research, Inc. | CRI |
| 7926678 | `HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.6782850599288941 | Couples Cure | Couples Therapy |
| 7926681 | `LOW_CONFIDENCE_TOP1` | 0.53 | Designalytics | Chainalytics |
| 7926694 | `LOW_CONFIDENCE_TOP1` | 0.53 | Entrokey Labs | Protocol Labs |
| 7926698 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Fenix Marine Services | FMS |
| 7926701 | `LOW_CONFIDENCE_TOP1` | 0.505137014056752 | Givebutter | BEEM |
| 7926704 | `WRONG_STATE_TOP1, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.8 | Illumina, Inc. (Primary) | Illumina, Inc. (Primary) |
| 7926707 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Jan Arnold LLC | JAL |
| 7926737 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.740607191707959 | Mammotome,  A Danaher Company | Danaher Tool Group |
| 7926742 | `LOW_CONFIDENCE_TOP1` | 0.527372939128155 | Merus | Mersen |
| 7926747 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | N6 | NB-620 |
| 7926749 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Next Horizon Leadership | NHL |
| 7926754 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | Oracle Corporation [PRIMARY] | Oracle America Inc |
| 7926758 | `LOW_CONFIDENCE_TOP1` | 0.53 | Parachute Health | Compass Health |
| 7926759 | `LOW_CONFIDENCE_TOP1` | 0.53 | Parallels | Parallels |
| 7926760 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pennington Partners & Co | Pennington Partners & Co |
| 7926762 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | Power Services Group | Power Team |
| 7926768 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Project Management Institute (PMI) [PRIMARY] | Project Management Institute (PMI) |
| 7926771 | `LOW_CONFIDENCE_TOP1` | 0.53 | Publicis Production | RUN PRODUCTION |
| 7926773 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8236403983151384 | QBE Insurance Crop Division | QBE Nau Country Insurance Company |
| 7926775 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | Raytheon Company [PRIMARY] | Raytheon IADC |
| 7926791 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spellers Freedom Foundation | Freedom Writers Foundation |
| 7926792 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9245805748325843 | Sterling Financial Group | Sterling Tours |
| 7926794 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Stryker [PRIMARY] | Stryker [PRIMARY] |
| 7926805 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.857505463070436 | Toyota Motor North America [PRIMARY] | Toyota Material Handling North America |
| 7926811 | `LOW_CONFIDENCE_TOP1` | 0.53 | Unrivaled Sports | SPORTS INC |
| 7926840 | `WRONG_STATE_TOP1, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.8072635084499573 | Travelmation | Travelmation |
| 7926845 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7419517143856476 | 2026 Annual Sales Meeting - Updated Dates KGNZK8T3DG6 | 113512 HPN - 2026 Annual Sales Meeting |
| 7926848 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7558189036414282 | 2026 WAF Strategic Offsite VKNPHHMHJYK | Strategic Offsite |
| 7926852 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | ABB 2027 X4NL2QTNT4H | A2X |
| 7926856 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | AE Perkins | Perkins & Will |
| 7926859 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | AKA | Also Known As |
| 7926861 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Aligned Data Systems | Tran Systems Advisors |
| 7926868 | `LOW_CONFIDENCE_TOP1` | 0.5232385478211012 | Anaplan Mini Connects-Seattle G2NVX28HBQ7 | Mobility Seattle V7N7YB66SLJ |
| 7926871 | `LOW_CONFIDENCE_TOP1` | 0.53 | As You Wish Coordination | Lang Coordination Services |
| 7926872 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Ashwood Construction LLC | ACL |
| 7926878 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8560601312750207 | Baylor Scott & White Med Ctr Irving | BAYLOR HEALTH CTR AT IRVING |

### Batch 13 (records 2400..2599)

- **In batch:** 200 | **Flagged:** 77 | **Flag rate:** 38.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7926881 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8164714707003349 | Board of Directors Meeting - AECOM - March FY27 2027 G5 | Board of Directors Meeting - AECOM - June 2022 FY22 ZSN |
| 7926896 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5185749327207622 | Canix | Onix |
| 7926897 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7870156561150474 | Carolina Complete Health Network | North Carolina Health Physics Society |
| 7926901 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Circle Internet Financial, LLC | Gateway Financial |
| 7926906 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | College of the Canyons Foundation | CCF |
| 7926910 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Couchbase | Couchbase |
| 7926911 | `LOW_CONFIDENCE_TOP1` | 0.53 | Covington Group | Covington Group |
| 7926914 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Data Networking Solutions | DNS |
| 7926915 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.765795524002272 | Dermody Properies 2027 Summit | The Global Summit 2027 |
| 7926922 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.5051962615355451 | Ellie's Event | Ellie's Meditative Movements |
| 7926924 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Enerflex | Enerflex |
| 7926926 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.74065553402557 | Eolian Employee Holdings | Eolian Energy |
| 7926928 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Executive Roundtable, LLC | Executive Roundtable, LLC |
| 7926929 | `LOW_CONFIDENCE_TOP1` | 0.53 | Exit Momentum | Momentum |
| 7926933 | `LOW_CONFIDENCE_TOP1` | 0.53 | FlatironDragados | Flatiron |
| 7926934 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Flipped Over Photos | FOP |
| 7926935 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7452586738642741 | Fondation Armand-Frappier | Fondation Ronald Denis |
| 7926941 | `LOW_CONFIDENCE_TOP1` | 0.53 | Front Runner Real Estate | ERA Select Real Estate |
| 7926943 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7604523265900164 | Global Talent Management Leadership Team Meeting LRN5JG | Talent Management Leaders |
| 7926945 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Goodwin Beckham | Mr. Peter Goodwin |
| 7926949 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | HBG | HB |
| 7926950 | `LOW_CONFIDENCE_TOP1` | 0.53 | HHHunt Corporation | HHHunt Corporation |
| 7926952 | `LOW_CONFIDENCE_TOP1` | 0.53 | HMHT-302 | HMT 302 |
| 7926954 | `LOW_CONFIDENCE_TOP1` | 0.5277095091592119 | Hotaling & Company | Technology Company |
| 7926961 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | International Society of Pharmacometrics | ISP |
| 7926963 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ITM USA, INC | Commend Usa |
| 7926964 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | IWMF | IWMF |
| 7926965 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Janne La | Show Go La |
| 7926967 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | John H Pitman High School | Pittsburg High School |
| 7926969 | `LOW_CONFIDENCE_TOP1` | 0.53 | Keenova Therapeutics | Merz Therapeutics |
| 7926975 | `LOW_CONFIDENCE_TOP1` | 0.53 | KFS Associates, LLC | KFS ASSOCIATES, LLC |
| 7926980 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Lindenwood Education System | LES |
| 7926984 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Luma Travel Co | LTC |
| 7926988 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Michael Rosenzweig | Michael Rosenzweig |
| 7926989 | `LOW_CONFIDENCE_TOP1` | 0.52 | Minne Group LA Rehearsals 2026 KBNH3CMHRFV | Minne Group LA Rehearsals 2026 KBNH3CMHRFV |
| 7926990 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Momentec Brands | Driven Brands |
| 7926991 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | MoveIn, Inc | Move Inc |
| 7926996 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NC Church of God of Prophecy | Mosaic Church |
| 7926997 | `LOW_CONFIDENCE_TOP1` | 0.53 | Nesnah Ventures | Neenah Enterprises |
| 7926998 | `LOW_CONFIDENCE_TOP1` | 0.53 | Nexiuum | Nexio |
| 7926999 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Nine In Texas LLc | NTL |
| 7927000 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7909664268775176 | Notre Dame American Society of Civil Engineers | Notre Dame |
| 7927001 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | November New Hire Training XFNCXNMNYW3 | New Hire Training Group 2026 |
| 7927003 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Oasis Beyond Travel and Tours | Oasis Tours |
| 7927005 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Offsite | Offsite '07 |
| 7927006 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Oliveda | Olive |
| 7927007 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Omar | Omar Ramos |
| 7927008 | `LOW_CONFIDENCE_TOP1` | 0.53 | On Location Events, LLC | On Location Events, LLC |
| 7927022 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Quick Quack Car Wash | Quick Quack Car Wash |
| 7927029 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Red Hat, Incorporated - HQ | Red Hat Red Hat, Incorporated - HQ |
| 7927030 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Reece | Reece |
| 7927033 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Retail Confectioners International | RCI |
| 7927038 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Rock Camp Alumni Foundation | RCAF |
| 7927047 | `LOW_CONFIDENCE_TOP1` | 0.5431814790191859 | SEA: United aircraft event GRN4ZXZBZF7 | Ultimate Aircraft |
| 7927048 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SEAOSC | SEAOSC |
| 7927050 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sevita Health | Sevita Health |
| 7927052 | `LOW_CONFIDENCE_TOP1` | 0.5369016988577614 | Sodexo Live! | SOI |
| 7927053 | `LOW_CONFIDENCE_TOP1` | 0.5480328000128274 | Soul Life Journeys | LifeNet |
| 7927058 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7752325267777229 | STM Sports & Collectibles, LLC | 3 Step Sports LLC |
| 7927059 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Strauss Dairy Ingredients | SDI |
| 7927061 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TBT Sports | SAVES Sports |
| 7927062 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Tennesseans for Student Success | TSS |
| 7927067 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.9194000000000001 | The Alumni Association of Tuskegee University, Incorpor | Tuskegee National Alumni Association 2019 |
| 7927068 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7668096866180674 | The Canadian Soccer Association | Canada Soccer |
| 7927070 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | The Detroit Traveling Crew | DTC |
| 7927071 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Nest School | Code School |
| 7927072 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The New Deal | The New Deal |
| 7927074 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7731587665245458 | TheaterWorks - Dog Man The Musical - Culver City, CA ZC | TheaterWorks - Dog Man The Musical - Portland, OR JJNPF |
| 7927075 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7696939775295105 | Tiger Aesthetics Recon Summit Z7NKPWPP7WJ | Tiger Aesthetics |
| 7927079 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TPG Global, LLC | Elementis Global LLC |
| 7927083 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8222723033597285 | United Real Estate Bluegrass | United Way of the Bluegrass |
| 7927086 | `LOW_CONFIDENCE_TOP1` | 0.53 | V&S Utilities | Public Utilities Commission |
| 7927091 | `LOW_CONFIDENCE_TOP1` | 0.53 | Verve Group | Verve Meetings & Events |
| 7927093 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vibe Booser Club | WHITE HOUSE NIGHT CLUB |
| 7927094 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Village Farms | Village Farms |
| 7927096 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7666865553837683 | Vistra Energy Board of Directors Meeting P6NJ53ZK77W | WWB May Board of Directors Meeting |
| 7927102 | `LOW_CONFIDENCE_TOP1` | 0.53 | West Paces | West Central Inc |

### Batch 14 (records 2600..2799)

- **In batch:** 200 | **Flagged:** 65 | **Flag rate:** 32.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7927106 | `LOW_CONFIDENCE_TOP1` | 0.4847136255881953 | Yetzirah | Tezrah |
| 7927108 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7654860375880648 | Zillow New Construction Forum 2026 ZQNFP7GFDTY | Zillow Group New Construction Summit Oct2017 |
| 7927126 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | EvolveCon | Evolve |
| 7927129 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Statement Movie, LLC | SML |
| 7927148 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Molly and Brian Auld | MBA |
| 7927154 | `LOW_CONFIDENCE_TOP1` | 0.53 | BMTCTN South East Consortium Spring Workshop | North East Florida Educational Consortium |
| 7927161 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Cycode | Cycode |
| 7927165 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Suncoast Food Brokerage | SFB |
| 7927169 | `LOW_CONFIDENCE_TOP1` | 0.53 | hanike | HANWHA |
| 7927170 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | hanike | Han |
| 7927171 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | HVBA | HVBA |
| 7927172 | `LOW_CONFIDENCE_TOP1` | 0.53 | Servius Group | Covius |
| 7927173 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tusker | Tusker Travels |
| 7927174 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Unitedhealthcare Student Resources | USR |
| 7927175 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | VMG IV group | VIG |
| 7927177 | `LOW_CONFIDENCE_TOP1` | 0.53 | Huckberry | Huckberry Offsite |
| 7927178 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Getaway with April May | Girl's Weekend Getaway |
| 7927181 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | BottomLine Development Group LLC | ahs Development Group LLC |
| 7927184 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8494749447295411 | 114876 HPN - 2026 RCNM Meeting | 112784 HPN - DC Meeting 2026 |
| 7927185 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Carrie Jo Coaching | CJC |
| 7927189 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Ian Verasammy | Ian Martin |
| 7927193 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Spencyr Mayer | The Mayer Brown Practices |
| 7927194 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Zoie Mortimer | MORTIMER & COMPANY CONSULTANTS |
| 7927196 | `LOW_CONFIDENCE_TOP1` | 0.5341592754686162 | Mother's Day Craft Fair 2026-  Kaua'i Made – Buy Kaua'i | SHOPO Hawaii |
| 7927202 | `LOW_CONFIDENCE_TOP1` | 0.53 | Primary Aimline Offsite Group | Offsite |
| 7927212 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7907524158580965 | American Dental Company | Society of American Indian Dentists |
| 7927220 | `LOW_CONFIDENCE_TOP1` | 0.53 | Laurel's Prospects 2026 | Emerge 2026 |
| 7927240 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Atlantic Claim Executives Associations | ACEA |
| 7927242 | `LOW_CONFIDENCE_TOP1` | 0.53 | Bonnie Christine | Christine Fazzi |
| 7927246 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8080843218119267 | Enabled Energy Company Meeting 2026 | Enabled Energy |
| 7927248 | `TOP5_TIE_CLUSTER` | 0.999 | Groups360 | Groups360 |
| 7927249 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Groups360 | Groups360 |
| 7927257 | `LOW_CONFIDENCE_TOP1` | 0.5427392317375228 | Natasia Lunford | Annemarie Cyboron |
| 7927258 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | New Destiny Church | NDC |
| 7927260 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7757404760183251 | Parsec Automation President's Club 2027 | Parsec Automation Corporation |
| 7927267 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Springline Advisory, LLC | SAL |
| 7927270 | `LOW_CONFIDENCE_TOP1` | 0.5484515459438118 | VTrips | CTRIP |
| 7927271 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Warrior Bride Ministries | WBM |
| 7927280 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7526890404663931 | International Co-Responder Alliance, Inc | Alliance |
| 7927286 | `LOW_CONFIDENCE_TOP1` | 0.53 | Box 5 Company LLC | Union Box Company |
| 7927294 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Julia & Sophia's Bridal Shower | Barbati Bridal Shower |
| 7927297 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | AZRA Games | Wavedash Games |
| 7927298 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Graham Healthcare | Graham Healthcare Group |
| 7927299 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | The Ultimate Brick Show | UBS |
| 7927300 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Iron City Cup | ICC |
| 7927302 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7519977557141082 | Baptist Communicators Association | Community Baptist Church |
| 7927309 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8080789339808391 | GEORGIA - PACIFIC CORPORATION | Pacific Communications |
| 7927320 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7613446805000106 | Eastern Kentucky Football @ Tarleton State X7N3ZW4BHG4 | Kennesaw State Football vs Eastern Kentucky Oct2021 |
| 7927324 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7636451699967144 | EPG Brand Acceleration Elevate Summit 2027 MKNLTYDMRV2 | EPG Brand Acceleration |
| 7927326 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8016268928223788 | Cru Summer Mission Briefing 2026 XQNY97RG29M | CCCI-Cru High School Summer Mission |
| 7927330 | `TOP5_TIE_CLUSTER` | 0.999 | Clover Health | Clover Health |
| 7927334 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Artivion AMDS Academy Nashville MVN98FNVPML | Nashville International Academy |
| 7927336 | `LOW_CONFIDENCE_TOP1` | 0.5423587292461818 | **CORRECTED** Chubb -  2026 Leadership Academy -  Dalla | 2023 Policy Leadership Conference |
| 7927344 | `LOW_CONFIDENCE_TOP1` | 0.5441532578671193 | 26 TUSA @ Alexandria & Springfield M3NNLL5ZFCQ | Alexandria Gibson |
| 7927345 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8350241484194164 | 26-0019 - Travel Portland L4NFLGMH6VK | Travel Portland |
| 7927346 | `LOW_CONFIDENCE_TOP1` | 0.5384665318045767 | China Tour Group Alexandria DVNP8KNV4XL | Elite Travelers Group |
| 7927348 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7470355610994206 | Billtrust Room Block X3NQM9RRX6Z | Luck Companies Room Block |
| 7927352 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | International Joint Commission (IJC) L3N6VNNSGL4 | International Joint Commission (IJC) L3N6VNNSGL4 |
| 7927355 | `LOW_CONFIDENCE_TOP1` | 0.5139544719837763 | Meeting and Hotel Group. Fortis Games. OCT 18 - 23, 202 | NRES Meeting on 17-18 Feb 2025 |
| 7927367 | `LOW_CONFIDENCE_TOP1` | 0.48823139950038574 | ID#20994-McKesson & Ontada: Together, Cancer Doesn't St | Stand Up To Cancer |
| 7927370 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7707081839426644 | SANS Nashville 2027 G7NMHM3HPP2 | SANS 2027 Conference |
| 7927375 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Online Bookings - SJCHM | OBS |
| 7927380 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | American Society for Cytology | ASC |
| 7927381 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Black Girl Ventures | Real Ventures |
| 7927382 | `LOW_CONFIDENCE_TOP1` | 0.53 | Center for International Blood & Marrow Transplant Rese | International Transplant Nurses Society NE Florida Chap |

### Batch 15 (records 2800..2999)

- **In batch:** 200 | **Flagged:** 73 | **Flag rate:** 36.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7927384 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Dairy Girl Network | DGN |
| 7927390 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | International Consortium on Governmental Financial Mana | International Consortium on Governmental Financial Mana |
| 7927391 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | College Broadcasters, Incorporated | CBI |
| 7927398 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | The Formerly Incarcerated Convicted People & Families M | The Formerly Incarcerated Convicted People & Families M |
| 7927399 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Virginia Elks Association | VEA |
| 7927408 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | NMA | NC Moves Assn |
| 7927412 | `LOW_CONFIDENCE_TOP1` | 0.4926638482375578 | Pacbag MJNM8JQ32NV | PACM |
| 7927421 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.800590113698807 | Utah Football @ TCU N8NXL8D36M7 | Colorado Football @ TCU VTN29WHTQG6 |
| 7927423 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7656186623650613 | Synergy Health Partners | Synergy |
| 7927424 | `LOW_CONFIDENCE_TOP1` | 0.53 | Secret Stuff | US SECRET SVC |
| 7927425 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bellami Professionals | GLI Professionals |
| 7927426 | `LOW_CONFIDENCE_TOP1` | 0.53 | Ferris Mowers | Ferris Travel Service |
| 7927427 | `TOP5_TIE_CLUSTER` | 0.9591999999999999 | Girls Inc | Girls Incorporated*** |
| 7927429 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Bruce Pittman, Incorporated | BPI |
| 7927431 | `LOW_CONFIDENCE_TOP1` | 0.52 | Military Flight Weekend Trip | Military Trip |
| 7927432 | `HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Newcome Wedding Services | NWS |
| 7927439 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Maritz – AT&L | Maritz – AT&L |
| 7927440 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | aBridgeview Sports Dome | ASD |
| 7927441 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | VisitPittsburgh | VisitPittsburgh |
| 7927444 | `LOW_CONFIDENCE_TOP1` | 0.53 | Intel Agents | Intel Security GSO Master |
| 7927446 | `LOW_CONFIDENCE_TOP1` | 0.53 | Holland Cousins Trip Sept 2026 | Holland Bucket List Trip |
| 7927454 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 113780 HPN - Sales Conference 2026 | 113780 HPN - Sales Conference 2026 |
| 7927463 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | Jack Morton OCCC | Confidential - Jack Morton |
| 7927464 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | OpenAI | OpenAI |
| 7927467 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Netflix, Inc.-United States | Netflix, Inc.-United States |
| 7927470 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | SPE | SP Paone Events |
| 7927471 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8655799143088635 | Tennessee and Mississippi Credit Union Association | Tennessee Credit Union League |
| 7927472 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Manufacturing Institute | EXCELLENCE IN MANUFACTURING CONSORTIUM |
| 7927473 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Texas Watermelon Association | TWA |
| 7927478 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Network Firm | The Firm |
| 7927479 | `LOW_CONFIDENCE_TOP1` | 0.5291021286107385 | Tesorio | Werner |
| 7927481 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | UCB, Inc. | UCB, Inc. |
| 7927487 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | VIO Med Spa | VMS |
| 7927488 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | SPR Pain Relief | SPR |
| 7927491 | `LOW_CONFIDENCE_TOP1` | 0.53 | SSI Strategy | SSI Strategy |
| 7927494 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Strategic Alliance | Strategic Alliances |
| 7927495 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Suja Life Juices | Suja Juice |
| 7927500 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Taysha Gene Therapies | TGT |
| 7927502 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Signature Healthcare | Signature Health Services |
| 7927506 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vizio | Vizio |
| 7927507 | `TOP5_TIE_CLUSTER` | 0.999 | Volaris Group | Volaris Group |
| 7927508 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Volvo Car USA LLC (SC Campus) | Volvo Car USA, LLC |
| 7927509 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Walk on Water | WOW |
| 7927512 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Westin | Westin |
| 7927513 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Whitecap Health Advisors | WHA |
| 7927516 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | WSB Finance | Western Finance Association |
| 7927517 | `TOP5_TIE_CLUSTER` | 0.999 | Xsem | XSEM |
| 7927518 | `LOW_CONFIDENCE_TOP1` | 0.53 | Yaskawa | Yaskawa |
| 7927529 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Nielsen IQ | Nielsen IQ |
| 7927531 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | One Triangle Resources | OTR |
| 7927534 | `LOW_CONFIDENCE_TOP1` | 0.53 | Pass Group | Passover Group |
| 7927546 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8462627590522294 | ICF INCORPORATED, LLC RESTON | ICF Incoporated, LLC |
| 7927547 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Image First | Image X |
| 7927548 | `LOW_CONFIDENCE_TOP1` | 0.48250301117067435 | Infinigate | Legalease |
| 7927551 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Integra Connect, LLC | ICL |
| 7927552 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Global Resilience Federation | GRF |
| 7927557 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Home Services of America | HSA |
| 7927562 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7585920621511496 | Johnson & Johnson Innovative Med | Johnson & Johnson MedTech |
| 7927564 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | LRN | LERN Resources Network |
| 7927567 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Merit Medical Systems | MMS |
| 7927569 | `LOW_CONFIDENCE_TOP1` | 0.53 | MH Live Events, LLC | K&D Global Events, LLC |
| 7927570 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7427847998350889 | Michelin (China) Investment Company Limited | Global Investment Company |
| 7927572 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | GEA Farm Tech | GFT |
| 7927574 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Gitlab | Gitlab |
| 7927577 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | F5 | Faze 5 |
| 7927578 | `LOW_CONFIDENCE_TOP1` | 0.53 | Fastest Labs | Century Labs |
| 7927579 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | First Tee Phoenix | FTP |
| 7927580 | `LOW_CONFIDENCE_TOP1` | 0.5231528296757979 | Fleetworthy | FleetOne |
| 7927588 | `LOW_CONFIDENCE_TOP1` | 0.53 | Crown Imports LLC dba Constellation Brands Beer Divisio | Crown Imports LLC dba Constellation Brands Beer Divisio |
| 7927589 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Consortium Affiliate - PepsiCo | CAP |
| 7927593 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | CFGI | CNO FINANCIAL GROUP, INC. |
| 7927595 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bohler | Bohler |
| 7927600 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | BD (Becton, Dickinson and Company) | BD  (Becton, Dickinson and Company) |

### Batch 16 (records 3000..3199)

- **In batch:** 200 | **Flagged:** 65 | **Flag rate:** 32.5%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7927602 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | ASCII Group, Incorporated | AGI |
| 7927604 | `LOW_CONFIDENCE_TOP1` | 0.5317412907275996 | Another Broken Egg of America Franchising, LLC | Steak Out Franchising INC |
| 7927605 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Apex Midwest | Midwest Events |
| 7927606 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ARC Cardinal | Arc Cardinal |
| 7927607 | `LOW_CONFIDENCE_TOP1` | 0.53 | Arizent | Arizent (formally SourceMedia) |
| 7927611 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | Ambleside Schools International | ASI |
| 7927612 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | AMDA | AM MEDICAL DIRECTORS ASSN |
| 7927614 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | American Cast Iron and Pipe Company | ACIPC |
| 7927615 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Abatement Technologies | CREATION TECHNOLOGIES |
| 7927618 | `LOW_CONFIDENCE_TOP1` | 0.5444004390436716 | ACDR | ACS |
| 7927628 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Petrone Associates | Petrone & Petrone |
| 7927629 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Freshworks Inc. | Freshworks, Inc. |
| 7927632 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Midland Resource Recovery, Inc. | Midland Resources Inc. |
| 7927633 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7502278962662701 | Evil Empire Speech Memorial Foundation | Nation Japanese American Memorial Foundation |
| 7927634 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Wellness Club | Grace Place Wellness |
| 7927636 | `LOW_CONFIDENCE_TOP1` | 0.5317277274180191 | Upstak | UPS - United Parcel Service |
| 7927643 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7825320785432404 | East Valley Pentacostal Church | Crown Valley Community Church |
| 7927650 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8043190961717452 | Fairlawn High School comp cheer team | Fairlawn High School Reunion |
| 7927652 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | TAG Global Travel | TGT |
| 7927654 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skynyd United Touring | VN Touring |
| 7927656 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TicketSpice | Tickets Now |
| 7927658 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7787366184616401 | Peak Technology Enterprises Inc. | Peak Projects LLC |
| 7927664 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.7833757019042968 | DAA | Dar Al Arqam |
| 7927666 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8497406313304267 | National Postal Mail Handlers Union AFL-CIO | American Postal Works Union, AFL-CIO |
| 7927668 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 10U Ashburn Shooting Stars | Ashburn Shooting Stars |
| 7927669 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 52817-HelmsBriscoe | HelmsBriscoe 2014 |
| 7927671 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 52817-HelmsBriscoe | 52817-HelmsBriscoe |
| 7927675 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.815725053836902 | Ashlee Nelson Curated Events | Curated by Ashlee |
| 7927676 | `TOP5_TIE_CLUSTER` | 0.7877505341401467 | CTM Meetings & Events North America - Do NOT Use | CTM Meetings & Events North America |
| 7927683 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF` | 0.79 | QVC | Q Vine Corporation |
| 7927690 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 6 PM Sports | 5430 Sports |
| 7927691 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Allergan Aesthetics | Allergan Aesthetics |
| 7927695 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | Colortokens | ColorTokens |
| 7927696 | `LOW_CONFIDENCE_TOP1` | 0.5207969465732832 | DistroKid | Codility |
| 7927700 | `LOW_CONFIDENCE_TOP1` | 0.5401569228686816 | FIT4MOM | Tech4Med |
| 7927703 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | GTIA | GTIA |
| 7927704 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1` | 0.52 | HealthSherpa | Healthsherpa |
| 7927710 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7672157042136558 | Shark Ninja APAC Holding Pte Ltd | Shark Ninja |
| 7927713 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Tracksuit Productions | Productions Plus |
| 7927714 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7507558328003049 | United Airlines Incorporated Singapore Branch | Singapore Airlines Staff Union |
| 7927715 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vet Vacation CE, Incorporated | Vet Vacation CE, Incorporated |
| 7927716 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2027 CPS Symposium | CPS |
| 7927731 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | Countdown City Classic | CCC |
| 7927733 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7741327287033415 | Departure Lounge Exchange - Oct 2027 | Departure Lounge |
| 7927743 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7741144155646777 | Fusion 360 degree | FUSION ACADEMY |
| 7927746 | `TOP5_TIE_CLUSTER` | 0.9592 | Groups360 | Groups360 |
| 7927773 | `LOW_CONFIDENCE_TOP1` | 0.5347812188858365 | IIAR - International Insitute of Ammonia Refrigeration | IOR The Institute of Refrigeration |
| 7927774 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | INMEX-Informed Meetings Exchange ASDC Q4 2026 Meeting | INMEX-Informed Meetings Exchange CONFIDENTIAL CLIENT -  |
| 7927776 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.75 | JetSet Modern Pilates | JMP |
| 7927808 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | NHS | N.C.H. Healthcare System |
| 7927809 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Nordic Pharma | PHARMA CI |
| 7927814 | `TOP5_TIE_CLUSTER` | 0.999 | Premier Dental | PREMIER DENTAL |
| 7927817 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Redbird Realty | Red Bird Travel |
| 7927824 | `TOP5_TIE_CLUSTER` | 0.999 | SMASHOUSE Creative Events Agency, Inc. | SMASHOUSE Creative Events Agency, Inc. |
| 7927827 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8131831331860098 | Star Mountain Fund Management | Star Mountain Capital |
| 7927829 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7756371380970497 | Strategic Association Management TxANA Annual Conferenc | Strategic Association Management |
| 7927831 | `LOW_CONFIDENCE_TOP1` | 0.5335539383357146 | Terranova Advising & Productions | NOVA |
| 7927833 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8661600074855323 | Texas Association of Behavioral Health Systems | North Texas Behavioral Health Authority |
| 7927834 | `TOP5_TIE_CLUSTER` | 0.999 | The Chefs Warehouse | The Chefs Warehouse |
| 7927835 | `TOP5_TIE_CLUSTER` | 0.999 | The Forum | The Forum |
| 7927836 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | TOSKANI | Toskani S.L. |
| 7927838 | `TOP5_TIE_CLUSTER` | 0.999 | traveling usa | Traveling USA |
| 7927842 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7837237201555811 | Workforce Acceleration Initiative | Workforce Velocity |
| 7927848 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | FROSCH | Frosch |
| 7927851 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | General Counsel AI | GCA |

### Batch 17 (records 3200..3216)

- **In batch:** 17 | **Flagged:** 6 | **Flag rate:** 35.3%

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7927854 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | General Counsel AI | GCA |
| 7927856 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION` | 0.69 | General Counsel AI | GCA |
| 7927857 | `LOW_CONFIDENCE_TOP1` | 0.53 | Global Travel Collection - Entertainment | Global Travel |
| 7927868 | `POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8523402080423562 | PCM Network 3rd party commiss | PCM Network |
| 7927869 | `TOP5_TIE_CLUSTER` | 0.999 | Prestige | Global Meeting Source | Prestige | Global Meeting Source |
| 7927871 | `TOP5_TIE_CLUSTER` | 0.999 | SMASHOUSE Creative Events Agency, Inc. | SMASHOUSE Creative Events Agency, Inc. |


## Priority: audit Gate A (`TOP5_TIE_CLUSTER`)

Review these first: all top-5 scores within **0.001** (same flag as `audit_scoring` Gate A).

| row_id | flags | score | query (trunc) | top-1 (trunc) |
|--------|-------|-------|-----------------|-----------------|
| 7879342 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | NWACUHO - Northwest Association of College & University | NW Association of College & University Housing Officers |
| 7880903 | `TOP5_TIE_CLUSTER` | 0.9746413510344029 | Corporation of Hamilton | Hamilton Enterprises |
| 7880906 | `TOP5_TIE_CLUSTER` | 0.9591999999999999 | Corporation Of Hamilton | HAMILTON GROUP |
| 7895819 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Lanthrop Gpm | GPM Life |
| 7913800 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | HB | H Beck |
| 7918803 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Wisconsin Evangelical Lutheran Church | WELS -Wisconsin Evangelical Lutheran Synod |
| 7919049 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | DLM Governance July 2026 | DMS Governance |
| 7919307 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | RHM Meeting | Personal Meeting |
| 7919311 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SAFTE-FAST User Conference | Fast Enterprise |
| 7919333 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7649896967635748 | RDB Hospitality -Rick Marino Group-  - Toronto | RDB Hospitality |
| 7919334 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | September Group | September Events |
| 7919451 | `TOP5_TIE_CLUSTER` | 0.999 | The Washington Institute | The Washington Institute |
| 7919683 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Annual Meeting of the Joint Society ASSCT | Annual Meeting |
| 7919704 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | OALA Training | Hands on Training |
| 7919706 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | SBCI Conferance | THE CONFERANCE BOARD |
| 7919755 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Schattastrum Industries | Omni Industries |
| 7919804 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spring 2026 District Meeting | Big Spring High School |
| 7919863 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Restaurant Strategy | Restaurant 365 |
| 7919911 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | VistaJet Ltd. | VistaJet Inc. |
| 7919912 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Portside Realty, LLC | Paramount Realty USA Llc. |
| 7919976 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Western Weather Group | Air Methods Western Region |
| 7920022 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pet Business Marketing Ltd | Advantage Business Marketing |
| 7920026 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Omega - Annual Client Appreciation | Alpha Omega Publishers |
| 7920128 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Whitmore Manufacturing LLC | Rim Manufacturing, LLC |
| 7920187 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SJCSH Online Bookings | Online Bookings -SFOBG |
| 7920198 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ServiceNow, Inc. | ServiceNow, Inc. |
| 7920202 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Pacific West Builders | Pacific Steel |
| 7920373 | `TOP5_TIE_CLUSTER` | 0.6790056870745864 | ONE VISION TALMA GROUP | Cancun | Jan 2026 | One Vision |
| 7920422 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ACIU North America 2025 National Convention | Universal North America |
| 7920475 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | American Ramallah Club of DC | American Ramallah Club NY |
| 7920510 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Wahlstrom / Gillis | Gillis |
| 7920521 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Winograd Group | Hope Winograd |
| 7920552 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 6th Air Naval Gunfire Liaison Co | 6th Air Naval Gunfire Liaison Co |
| 7920581 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026-Gunslinger Ohio State | Ohio National |
| 7920654 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | San Lorenzo Unified School District | San Lorenzo Unified School District |
| 7920661 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | VIEWS FC | Cap FC United |
| 7920696 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Power Stream Fitness | Fluid Power Resource |
| 7920714 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Waugh/Mayer | Waugh & Co, Inc |
| 7920734 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | PLI Pastoral Leadership Institute | Academy of Religious Leadership |
| 7920825 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 21st Annual Government-to-Government Tribal Consultatio | Sycuan Tribal Government Office |
| 7920874 | `TOP5_TIE_CLUSTER` | 0.999 | The Washington Commanders | The Washington Commanders |
| 7920875 | `TOP5_TIE_CLUSTER` | 0.999 | SALESFORCE INC. | Salesforce Inc. |
| 7920877 | `TOP5_TIE_CLUSTER` | 0.999 | Salesforce - Primary | Salesforce - Primary |
| 7920972 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Saloon/Lodge Events 2026 | Yellowhouse Events |
| 7921255 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SANCC Restaurant Outlet Bookings | SANCC Restaurant Outlet Bookings |
| 7921274 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Novo | Novo |
| 7921303 | `TOP5_TIE_CLUSTER` | 0.5997711601204783 | Sendflow | Perflow |
| 7921305 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SLB | SLB |
| 7921341 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | PG Solutions | PG SOLUTIONS |
| 7921361 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Volvo Construction Equipment Haulers | Volvo Construction Equipment AB |
| 7921379 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | WBD - TNTLA | WBD - TNTLA |
| 7921435 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Paceline Equity Partners | Partners Financial |
| 7921488 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Yoshoku Q2 2026 | Yoshoku Events 2025 |
| 7921663 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Randstad Enterprise | Randstad Digital, LLC. |
| 7921689 | `TOP5_TIE_CLUSTER` | 0.999 | Society For Investigative Derma | Society For Investigative Derma |
| 7921690 | `TOP5_TIE_CLUSTER` | 0.7685551206804037 | PESTOLA 2026 - Affiliates | 2026 ACC Affiliates |
| 7921770 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Schwartz and Company | Gregory J Schwartz & Company |
| 7921774 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | YA Group | YA Group |
| 7921827 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Smurfit Westrock | Smurfit Westrock |
| 7922143 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Rawden Joint Ventures | Red Ventures Recruiting |
| 7922160 | `TOP5_TIE_CLUSTER` | 0.999 | SLCCC Small Meeting | SLCCC Small Meeting |
| 7922167 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922173 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922275 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7922276 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | VitalCare Infusion Services | VitalCare Infusion Services |
| 7922279 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Walton Ward Accommodations | Pastner Room Accommodations |
| 7922388 | `TOP5_TIE_CLUSTER` | 0.8497436655752315 | SPSA 2026 Affiliates | 2026 ACC Affiliates |
| 7922404 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Warrior Kido | Live Free Warrior |
| 7922482 | `TOP5_TIE_CLUSTER` | 0.8995 | Viviana Viajes S.A. de C.V. | Viajes Lorimar S.A. de C.V. |
| 7922695 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pandora Y Flans | Pandora Y Flans |
| 7922760 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Scouting America | National Football Scouting |
| 7922828 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YMCA of the North | YMCA of the North |
| 7922856 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NINE21 Productions | Dance America Productions |
| 7922865 | `TOP5_TIE_CLUSTER` | 0.8057561942584268 | Nteractive | NTERACTIVE |
| 7923092 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sheriff's Employees' Benefit Association (SEBA) | Sheriff's Employees' Benefit Association (SEBA) |
| 7923095 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Stateline Road | Road to California |
| 7923117 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TPC Management | TPC Transaction Processing Performance Council |
| 7923119 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tynan Group | Tynan Group |
| 7923129 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | WKS USA | W Standard USA |
| 7923210 | `TOP5_TIE_CLUSTER` | 0.8 | Wedaways Travel | Wedaways Travel |
| 7923252 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | St. Christopher Catholic Parish | St. Patrick Catholic Community |
| 7923253 | `TOP5_TIE_CLUSTER` | 0.999 | TD SYNNEX Corporation | TD SYNNEX Corporation |
| 7923363 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Youth Baseball Team / Westlake Village Gladiators | Westridge Ranchers Baseball Team |
| 7923366 | `TOP5_TIE_CLUSTER` | 0.999 | TAG - LA | TAG - LA |
| 7923432 | `TOP5_TIE_CLUSTER` | 0.999 | The Westfield Group A | Westfield Group |
| 7923439 | `TOP5_TIE_CLUSTER` | 0.999 | UATP | UATP |
| 7923445 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Westiminster School | Unquowa School |
| 7923606 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 108 Coaching Ltd | Focal Point Coaching |
| 7923628 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | HBA-Healthcare Businesswomen's Association | HBA Healthcare Businesswomens Association |
| 7923657 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | TAM Advisory | ATX Advisory |
| 7923804 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | n3xt | NXTP |
| 7923830 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Timothy Initiative | United Religious Initiative |
| 7923831 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tik Tok | Tik Tok |
| 7923832 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travel Event Staffing | Event Travel Management, N.A. |
| 7923968 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | La Boite Rouge Vif | Bleu Blanc Rouge |
| 7924004 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Ahmad and Susan | Susan Fong |
| 7924032 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | SMRF - Individual | SMRF - Individual |
| 7924066 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Smithfield Selma High School | Smithfield Selma High School |
| 7924088 | `TOP5_TIE_CLUSTER` | 0.6654991631705935 | T2C Sports - Wilmington Academy of Arts and Sciences (m | Academy of Arts & Sciences |
| 7924102 | `TOP5_TIE_CLUSTER` | 0.9040344149903817 | Berkshire Conference of Women Historians 2029 | Berkshire Conference of Women Historians |
| 7924123 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Antioch Church | Antioch Community Church |
| 7924124 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7834226259818444 | ALG Vacations/2026  Amao70Getaway / Cancun | ALG Vacations |
| 7924130 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | A.R.T. Global Education Ltd | CIES Comparative & International Education Society |
| 7924163 | `TOP5_TIE_CLUSTER` | 0.8995 | Toronto Request -Worldimension | Montreal Request-Worldimension |
| 7924174 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Sportograf Digital Solutions GMBH | Digital Art Solutions |
| 7924187 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Smithfield-Selma High School | Smithfield-Selma High School |
| 7924225 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | Th | Terese Hoogoian |
| 7924240 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Royalton Hotels and Resorts | Royal Resorts |
| 7924301 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Affinity | AFFINITY TOURS |
| 7924310 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 ASCO Affiliates | 2026 ACC Affiliates |
| 7924427 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sedona Star Holistic Spa | Spa Nautica |
| 7924435 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Travora Global LLP | Travora Global LLP |
| 7924436 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TD Bank | TD Bank |
| 7924456 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | IEA Interscholastic Equestrian Association | University of Findlay Equestrian |
| 7924494 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Americain Society of Questioned Document Examiners | Southwestern Association of Forensic Document Examiners |
| 7924531 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | All Star Life Group | All Star Group |
| 7924593 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.7468751123973302 | ALG Vacations / Saasha Day / Cancun | ALG Vacations |
| 7924650 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travelicious Tours Pvt. Limited | Travelicious Tours PVT Limited |
| 7924652 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Solventum | Solventum |
| 7924699 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Apex Agency | APEX Agency |
| 7924713 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | UKG Inc. | UKG Inc. |
| 7924770 | `TOP5_TIE_CLUSTER` | 0.9194000000000001 | Banorte Wealth Management | CBG Wealth Management |
| 7924799 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Waypoint, LLC | CrossPointe, LLC |
| 7924858 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Adept Fasteners | Adept Technologies |
| 7924915 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | YYZMO Online Bookings | YYZMO Online Bookings |
| 7924942 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Ahava International | Voyage International |
| 7924957 | `TOP5_TIE_CLUSTER` | 0.7955893337726593 | Stribe Dental | Dental Hygiene |
| 7924976 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bonita Trip | Bonita Travels |
| 7924992 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | World Tour Logistics LLC | Impact Logistics |
| 7925020 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Thermomix | Thermomix |
| 7925030 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Working Minds | Construction Working Minds |
| 7925040 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Brook Health | Brook Health |
| 7925048 | `TOP5_TIE_CLUSTER` | 0.999 | Alliant Healthcare Solutions | Alliant Healthcare Solutions |
| 7925054 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skills / Compétences Canada | Skills / Compétences Canada |
| 7925113 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | Woodbrey Family Travel | Family and Friends Travel |
| 7925118 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Shionogi Inc | Shionogi Inc |
| 7925299 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | General HVAC Solutions America, Inc. | General HVAC Solutions America, Inc. |
| 7925337 | `TOP5_TIE_CLUSTER` | 0.999 | PSI CRO | PSI CRO |
| 7925352 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Waters Corporation | Waters Corporation |
| 7925380 | `TOP5_TIE_CLUSTER` | 0.999 | Accuity | Accuity |
| 7925393 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Grindr | Grindr |
| 7925396 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Me Plus Ultra | Material Plus |
| 7925437 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | CATHEDRAL OF FAITH FAMILY PRAISE CENTER INTERNATIONAL | International Cathedral of Faith Fellowship |
| 7925440 | `TOP5_TIE_CLUSTER` | 0.999 | Good Feet Store | The Good Feet Store |
| 7925442 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Kingston MYRES - 2024 | MYRES Wedding 2022 |
| 7925491 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | New Greater Deliverance Church | House of Deliverance Church |
| 7925492 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | New Greater Deliverance Church | Next Level Church |
| 7925495 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | NPS Panels | NPS Air Resources Division |
| 7925509 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 North American Sales Meeting | NA Sales Meeting Jan 2022 |
| 7925513 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | America’s Auto Auction | America's Auto Auction |
| 7925549 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Constellatin Brands | Best Life Brands |
| 7925588 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Illume Events | On Points Events |
| 7925595 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Katie Brown Educational Program | EDUCATIONAL PROGRAMS |
| 7925604 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Lunamare Escapes | PORTER ESCAPES INC. |
| 7925610 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | McCarthy Holdings | McCarthy Holdings, Inc. |
| 7925615 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | naturaLED | Natural High |
| 7925616 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Neoss | NEOS |
| 7925620 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NorthStar Insurance Services, Incorporated | NorthStar Insurance Services, Incorporated |
| 7925629 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | OGT | ogt |
| 7925694 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sojourn Adventures | Discover   Adventures |
| 7925697 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spellers Freedom Foundation | Freedom Writers Foundation |
| 7925726 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Towie Travel | Great Adventure Travel |
| 7925768 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Institute of Scrap Recycling Industries d/b/a Recycled  | Institute of Scrap Recycling Industries, Gulf Coas |
| 7925804 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Henry Linda | Linda Henry |
| 7925838 | `TOP5_TIE_CLUSTER` | 0.8796000000000002 | Online Bookings - WASSL | Online Bookings - RLSC |
| 7925863 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | JOICY NAILS SPA | Shine Massage and Spa |
| 7925868 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Marie Moran & Company, LLC | Marie Moran & Company, LLC |
| 7925876 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Morven Park | Morven Park |
| 7925896 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Trump National Golf Club Washington DC LLC | Trump National Golf Club Washington DC LLC |
| 7925900 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | T.White Parker | T.White Parker |
| 7925919 | `TOP5_TIE_CLUSTER` | 0.999 | SFK Tours | SFK Tours |
| 7925951 | `TOP5_TIE_CLUSTER` | 0.8995 | 2026 Worldwide Sales Kick Off Meeting | 2023 Genie Sales Kick-Off Meeting |
| 7925962 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Howden Re | Howden Re |
| 7925965 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Financial Services Incentive Program | 2003 Qwest Incentive Program |
| 7926071 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Jabra Ghanayem Engagement Party RB | Vora Engagement Party |
| 7926108 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Recruitment Coach | Beyond Being Coach |
| 7926171 | `TOP5_TIE_CLUSTER` | 0.8995 | Terns Ad Board | March Ad Board |
| 7926201 | `TOP5_TIE_CLUSTER` | 0.999 | VF Corp | VF CORP |
| 7926234 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Raffles and Fairmont | Raffles and Fairmont |
| 7926235 | `TOP5_TIE_CLUSTER` | 0.999 | Paymentus | Paymentus |
| 7926242 | `TOP5_TIE_CLUSTER` | 0.6159726865780956 | PharmOne | Pharma |
| 7926259 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Netwrix | Netwrix |
| 7926261 | `TOP5_TIE_CLUSTER` | 0.999 | Newport Group | Newport Group |
| 7926269 | `TOP5_TIE_CLUSTER` | 0.8995 | Mercury Marketing Team Offsite | Replicated Marketing Team Offsite |
| 7926271 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Management Off-site | Management Committee Off-site and Teambuilding |
| 7926285 | `TOP5_TIE_CLUSTER` | 0.999 | National Alliance of Wound Care & Ostomy | National Alliance of Wound Care & Ostomy |
| 7926297 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | KENX | KENX |
| 7926340 | `TOP5_TIE_CLUSTER` | 0.8995 | Banner Partner Retreat | 2024 Partner Retreat |
| 7926348 | `TOP5_TIE_CLUSTER` | 0.999 | Baker Tilly Advisory Group, LP | Baker Tilly Advisory Group, LP |
| 7926351 | `TOP5_TIE_CLUSTER` | 0.9492499999999999 | Annual Conference 2026 | 2026 COSA Annual Conference |
| 7926354 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Annual Women’s Health Conference | U.S. Women’s Health Alliance |
| 7926358 | `TOP5_TIE_CLUSTER` | 0.999 | Ariat International | Ariat International |
| 7926368 | `TOP5_TIE_CLUSTER` | 0.8995 | Board Meetings ID1241165 | Advisory Board Meetings |
| 7926453 | `TOP5_TIE_CLUSTER, POSSIBLE_ENTITY_TAIL_MISMATCH` | 0.8995 | Customer Advisory Board (CAB) 2026 | Evotek 2025 Customer Advisory Board |
| 7926485 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Affiliated Independent Distributors, Inc | Affiliated Independent Distributors, Inc |
| 7926487 | `TOP5_TIE_CLUSTER` | 0.999 | Bayer Meeting Management Team @ Maritz T | Bayer Meeting Management Team @ Maritz T |
| 7926488 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Bayer Meeting Management Team @ Maritz T | Bayer Meeting Management Team @ Maritz T |
| 7926514 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skirt | Skirt |
| 7926515 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Meeting Advocate | Advocate |
| 7926516 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Optimus Travel | Destiny Travel |
| 7926517 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Travel by JB | JB Travel |
| 7926539 | `TOP5_TIE_CLUSTER` | 0.999 | Groups360 | Groups360 |
| 7926546 | `TOP5_TIE_CLUSTER` | 0.999 | Progress Residential | Progress Residential |
| 7926550 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2026 ND Golf Trip | Total Golf Travel |
| 7926556 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Baker & Karvala Ceremony | Slivka / Baker Wedding |
| 7926565 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Databricks Inc. | Databricks Inc. |
| 7926569 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | EPIC 21 | Century 21 Moves |
| 7926577 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | James Wenk Outing | Pelkey Golf Outing |
| 7926608 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Third Cost Wealth Advisors | Independent Financial Advisors |
| 7926609 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Thrivent | Thrivent |
| 7926613 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vivian Calusinski Yoga | Yoga Six |
| 7926645 | `TOP5_TIE_CLUSTER` | 0.999 | Agility Fuel Solutions | Agility Fuel Solutions |
| 7926647 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Allocate | Allocate |
| 7926669 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | CMX | CMS |
| 7926760 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Pennington Partners & Co | Pennington Partners & Co |
| 7926768 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Project Management Institute (PMI) [PRIMARY] | Project Management Institute (PMI) |
| 7926791 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Spellers Freedom Foundation | Freedom Writers Foundation |
| 7926794 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Stryker [PRIMARY] | Stryker [PRIMARY] |
| 7926856 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | AE Perkins | Perkins & Will |
| 7926861 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Aligned Data Systems | Tran Systems Advisors |
| 7926901 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Circle Internet Financial, LLC | Gateway Financial |
| 7926910 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Couchbase | Couchbase |
| 7926924 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Enerflex | Enerflex |
| 7926928 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Executive Roundtable, LLC | Executive Roundtable, LLC |
| 7926945 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Goodwin Beckham | Mr. Peter Goodwin |
| 7926949 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | HBG | HB |
| 7926963 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ITM USA, INC | Commend Usa |
| 7926964 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | IWMF | IWMF |
| 7926965 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Janne La | Show Go La |
| 7926967 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | John H Pitman High School | Pittsburg High School |
| 7926988 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Michael Rosenzweig | Michael Rosenzweig |
| 7926990 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Momentec Brands | Driven Brands |
| 7926996 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | NC Church of God of Prophecy | Mosaic Church |
| 7927001 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | November New Hire Training XFNCXNMNYW3 | New Hire Training Group 2026 |
| 7927003 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Oasis Beyond Travel and Tours | Oasis Tours |
| 7927005 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Offsite | Offsite '07 |
| 7927007 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Omar | Omar Ramos |
| 7927022 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Quick Quack Car Wash | Quick Quack Car Wash |
| 7927029 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Red Hat, Incorporated - HQ | Red Hat Red Hat, Incorporated - HQ |
| 7927030 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Reece | Reece |
| 7927048 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | SEAOSC | SEAOSC |
| 7927050 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Sevita Health | Sevita Health |
| 7927061 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TBT Sports | SAVES Sports |
| 7927071 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Nest School | Code School |
| 7927072 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The New Deal | The New Deal |
| 7927079 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TPG Global, LLC | Elementis Global LLC |
| 7927093 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vibe Booser Club | WHITE HOUSE NIGHT CLUB |
| 7927094 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Village Farms | Village Farms |
| 7927126 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | EvolveCon | Evolve |
| 7927170 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | hanike | Han |
| 7927173 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Tusker | Tusker Travels |
| 7927178 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Getaway with April May | Girl's Weekend Getaway |
| 7927181 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | BottomLine Development Group LLC | ahs Development Group LLC |
| 7927189 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Ian Verasammy | Ian Martin |
| 7927193 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Spencyr Mayer | The Mayer Brown Practices |
| 7927194 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Zoie Mortimer | MORTIMER & COMPANY CONSULTANTS |
| 7927248 | `TOP5_TIE_CLUSTER` | 0.999 | Groups360 | Groups360 |
| 7927249 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Groups360 | Groups360 |
| 7927294 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Julia & Sophia's Bridal Shower | Barbati Bridal Shower |
| 7927297 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | AZRA Games | Wavedash Games |
| 7927298 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Graham Healthcare | Graham Healthcare Group |
| 7927330 | `TOP5_TIE_CLUSTER` | 0.999 | Clover Health | Clover Health |
| 7927334 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Artivion AMDS Academy Nashville MVN98FNVPML | Nashville International Academy |
| 7927352 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | International Joint Commission (IJC) L3N6VNNSGL4 | International Joint Commission (IJC) L3N6VNNSGL4 |
| 7927381 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Black Girl Ventures | Real Ventures |
| 7927390 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | International Consortium on Governmental Financial Mana | International Consortium on Governmental Financial Mana |
| 7927398 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | The Formerly Incarcerated Convicted People & Families M | The Formerly Incarcerated Convicted People & Families M |
| 7927425 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bellami Professionals | GLI Professionals |
| 7927427 | `TOP5_TIE_CLUSTER` | 0.9591999999999999 | Girls Inc | Girls Incorporated*** |
| 7927439 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Maritz – AT&L | Maritz – AT&L |
| 7927441 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | VisitPittsburgh | VisitPittsburgh |
| 7927454 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 113780 HPN - Sales Conference 2026 | 113780 HPN - Sales Conference 2026 |
| 7927463 | `TOP5_TIE_CLUSTER` | 0.9046805748325841 | Jack Morton OCCC | Confidential - Jack Morton |
| 7927464 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | OpenAI | OpenAI |
| 7927467 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Netflix, Inc.-United States | Netflix, Inc.-United States |
| 7927472 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Manufacturing Institute | EXCELLENCE IN MANUFACTURING CONSORTIUM |
| 7927478 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Network Firm | The Firm |
| 7927481 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | UCB, Inc. | UCB, Inc. |
| 7927494 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Strategic Alliance | Strategic Alliances |
| 7927495 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Suja Life Juices | Suja Juice |
| 7927502 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Signature Healthcare | Signature Health Services |
| 7927506 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vizio | Vizio |
| 7927507 | `TOP5_TIE_CLUSTER` | 0.999 | Volaris Group | Volaris Group |
| 7927508 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Volvo Car USA LLC (SC Campus) | Volvo Car USA, LLC |
| 7927512 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Westin | Westin |
| 7927516 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | WSB Finance | Western Finance Association |
| 7927517 | `TOP5_TIE_CLUSTER` | 0.999 | Xsem | XSEM |
| 7927529 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Nielsen IQ | Nielsen IQ |
| 7927547 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Image First | Image X |
| 7927574 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Gitlab | Gitlab |
| 7927577 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | F5 | Faze 5 |
| 7927595 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Bohler | Bohler |
| 7927600 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | BD (Becton, Dickinson and Company) | BD  (Becton, Dickinson and Company) |
| 7927605 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Apex Midwest | Midwest Events |
| 7927606 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | ARC Cardinal | Arc Cardinal |
| 7927615 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Abatement Technologies | CREATION TECHNOLOGIES |
| 7927628 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Petrone Associates | Petrone & Petrone |
| 7927629 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Freshworks Inc. | Freshworks, Inc. |
| 7927632 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Midland Resource Recovery, Inc. | Midland Resources Inc. |
| 7927634 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | The Wellness Club | Grace Place Wellness |
| 7927654 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | Skynyd United Touring | VN Touring |
| 7927656 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | TicketSpice | Tickets Now |
| 7927668 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 10U Ashburn Shooting Stars | Ashburn Shooting Stars |
| 7927669 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 52817-HelmsBriscoe | HelmsBriscoe 2014 |
| 7927671 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 52817-HelmsBriscoe | 52817-HelmsBriscoe |
| 7927676 | `TOP5_TIE_CLUSTER` | 0.7877505341401467 | CTM Meetings & Events North America - Do NOT Use | CTM Meetings & Events North America |
| 7927690 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | 6 PM Sports | 5430 Sports |
| 7927691 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Allergan Aesthetics | Allergan Aesthetics |
| 7927713 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Tracksuit Productions | Productions Plus |
| 7927715 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Vet Vacation CE, Incorporated | Vet Vacation CE, Incorporated |
| 7927716 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | 2027 CPS Symposium | CPS |
| 7927746 | `TOP5_TIE_CLUSTER` | 0.9592 | Groups360 | Groups360 |
| 7927808 | `QUERY_HAS_GEO_TOP1_BARE_ROW, HIGH_SCORE_NEAR_ZERO_LOCATION, MATCH_TYPE_LOW_CONF, TOP5_TIE_CLUSTER` | 0.79 | NHS | N.C.H. Healthcare System |
| 7927809 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Nordic Pharma | PHARMA CI |
| 7927814 | `TOP5_TIE_CLUSTER` | 0.999 | Premier Dental | PREMIER DENTAL |
| 7927817 | `LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.53 | Redbird Realty | Red Bird Travel |
| 7927824 | `TOP5_TIE_CLUSTER` | 0.999 | SMASHOUSE Creative Events Agency, Inc. | SMASHOUSE Creative Events Agency, Inc. |
| 7927834 | `TOP5_TIE_CLUSTER` | 0.999 | The Chefs Warehouse | The Chefs Warehouse |
| 7927835 | `TOP5_TIE_CLUSTER` | 0.999 | The Forum | The Forum |
| 7927836 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | TOSKANI | Toskani S.L. |
| 7927838 | `TOP5_TIE_CLUSTER` | 0.999 | traveling usa | Traveling USA |
| 7927848 | `WRONG_STATE_TOP1, LOW_CONFIDENCE_TOP1, TOP5_TIE_CLUSTER` | 0.52 | FROSCH | Frosch |
| 7927869 | `TOP5_TIE_CLUSTER` | 0.999 | Prestige | Global Meeting Source | Prestige | Global Meeting Source |
| 7927871 | `TOP5_TIE_CLUSTER` | 0.999 | SMASHOUSE Creative Events Agency, Inc. | SMASHOUSE Creative Events Agency, Inc. |


**Run `plan-no-geo-exact-20260430` summary:** 3217 records, **1065** flagged at least once (33.1% of rows).
