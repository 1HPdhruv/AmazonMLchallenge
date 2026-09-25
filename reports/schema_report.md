# Schema report

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

- entities (Source 1): 6000
- records: 12167 {'source2': 6711, 'source3': 5456}

## Fields present per source
- source1: entity_id, business_name, business_address, country, (parsed from business_address: street/city/region/postal)
- source2: entity_id, business_name, business_address, country, (parsed from business_address: street/city/region/postal)
- source3: entity_id, business_name, business_address, country, (parsed from business_address: street/city/region/postal)

## Countries (open set)
- Source 1: {'US': 3340, 'India': 2660}
- records: {'US': 6804, 'India': 5363}

## Field completeness (Source 1)
| field | non-null | unique |
|---|---|---|
| entity_id | 1.0 | 6000 |
| name | 1.0 | 5245 |
| address | 0.97 | 5256 |
| city | 0.97 | 14 |
| region | 0.97 | 12 |
| postal | 0.9223 | 168 |
| country | 1.0 | 2 |
| full_address | 0.97 | 5763 |

## Field completeness (records)
| field | non-null | unique |
|---|---|---|
| record_id | 1.0 | 12167 |
| source | 1.0 | 2 |
| name | 0.9687 | 11285 |
| address | 0.8992 | 9766 |
| city | 0.7654 | 533 |
| region | 0.7063 | 56 |
| postal | 0.6586 | 195 |
| country | 1.0 | 2 |
| full_address | 0.8992 | 10771 |

## Labels
- n_label_pairs: 5952
- records_linked_to_multiple_entities: 0
- entities_no_match: 2102
- entities_single_match: 2731
- entities_multi_match: 1167
- max_matches_per_entity: 4

## Warnings
- records: 45 rows with neither name nor address
