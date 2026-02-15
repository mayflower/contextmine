# QA Checklist: Extracted Views im Browser (pro Projekt/Collection)

## Zweck
Diese Checkliste validiert, dass die extrahierten Readonly-Sichten im Cockpit korrekt aus dem Twin pro Collection geladen werden.

## Voraussetzungen
1. Mindestens zwei Collections mit Datenbestand.
2. Für mindestens eine Collection existiert ein AS-IS-Szenario.
3. Optional: Ein TO-BE-Szenario mit `base_scenario_id` für den Mermaid-Vergleich.

## Test 1: Projekt-Isolation
1. Cockpit öffnen.
2. Collection A wählen und sichtbare Daten merken (Szenarien, Graph, Hotspots).
3. Auf Collection B wechseln.
4. Erwartung:
   - Inhalte wechseln sichtbar.
   - Keine Daten aus Collection A bleiben stehen.

## Test 2: Layer-Filter in Topology/Deep Dive
1. Tab `Topology` öffnen.
2. Layer nacheinander umschalten:
   - `portfolio_system`
   - `domain_container`
   - `component_interface`
   - `code_controlflow`
3. Erwartung:
   - Knoten/Kanten ändern sich je Layer.
   - Keine Fehlermeldung im Cockpit.

## Test 3: City-Sicht Konsistenz
1. Tab `City` öffnen.
2. Prüfen, ob `Metric Nodes`, `Avg Coverage`, `Avg Complexity`, `Avg Coupling` angezeigt werden.
3. In DevTools die Response von `GET /api/twin/collections/{collection_id}/views/city` prüfen.
4. Erwartung:
   - `summary.metric_nodes` entspricht `cc_json.nodes.length`.
   - `hotspots` ist bei nicht-leerer Collection nicht leer.

## Test 4: Mermaid AS-IS vs TO-BE
1. AS-IS-Szenario wählen, Tab `Mermaid C4`.
2. TO-BE-Szenario wählen (mit `base_scenario_id`), Tab `Mermaid C4`.
3. Erwartung:
   - AS-IS: Single-Ausgabe.
   - TO-BE: Vergleichsausgabe mit zwei Blöcken (`AS-IS` und `TO-BE`).

## Test 5: Exportformate Smoke Test
1. Tab `Exporte` öffnen.
2. Jeweils Export erzeugen für:
   - `cc_json`
   - `cx2`
   - `jgf`
   - `lpg_jsonl`
   - `mermaid_c4`
3. Erwartung:
   - `cc_json`: enthält `projectName`, `nodes`, `edges`.
   - `cx2`: enthält `CXVersion`.
   - `jgf`: enthält `graph`.
   - `lpg_jsonl`: enthält JSONL-Zeilen mit `type=node|edge`.
   - `mermaid_c4`: enthält `C4Container`.

## API-Referenz (für Debug)
1. `GET /api/twin/collections/{collection_id}/views/topology`
2. `GET /api/twin/collections/{collection_id}/views/deep-dive`
3. `GET /api/twin/collections/{collection_id}/views/city`
4. `GET /api/twin/collections/{collection_id}/views/mermaid`
5. `POST /api/twin/scenarios/{scenario_id}/exports`
6. `GET /api/twin/scenarios/{scenario_id}/exports/{export_id}`
