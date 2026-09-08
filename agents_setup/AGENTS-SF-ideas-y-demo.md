# AgentsAI aplicado a Surrogate Factory (VS Code) — ideas y demo "Example - Marta"

Encaja con la estrategia del PDF: VS Code Chat + modelos DIV de DAISEI, skills en `.github/skills/`, agentes custom en `.github/agents/` (mismo patrón que el ejemplo streamlit-ui y los agentes coder/reviewer/tester).

## Qué he preparado (listo para el repo Pipelines)

| Archivo | Qué hace |
|---|---|
| `.github/skills/sf-pipeline/SKILL.md` | El "blueprint" de SF: layout UCxxx, formas exactas de pipeline_config.yaml y los YAML SF_1..SF_9 (sacadas de UCHardLanding), reglas de elección de modelo aprendidas de los 5 UCs, cómo ejecutar (run_pipeline.py) y cómo leer el veredicto del report. |
| `.github/agents/pipeline-builder.agent.md` | Agente que construye un UC completo: entrevista → inspecciona el CSV → elige campeón/baseline con justificación → scaffolding copiando UCHardLanding → smoke run → run de producción → tabla de veredicto Q90 con acción correctiva por FAIL. |
| `.github/agents/validation-interpreter.agent.md` | Agente de solo lectura: lee el executive summary de un UC y devuelve los 4 bloques (split, accuracy, bias, coverage) con la acción del playbook por cada check en rojo. |

Con los archivos en el repo, en VS Code Chat aparecen como `@pipeline-builder` y `@validation-interpreter` (Configure Custom Agents), y la skill se aplica sola cuando la petición trata de pipelines.

## La demo para tu sección (08 · Example - Marta)

Objetivo del ejemplo: "un agente que te construye una pipeline". Guion:

1. Punto de partida: solo un CSV y unos requisitos. Sugerencia reproducible: `UCHardLanding/data/datos.csv` renombrado como caso nuevo (UCDemo), porque existe, es pequeño y GB entrena en minutos. Alternativa más vistosa: un CSV nuevo que te pasen de otra disciplina.
2. Prompt a `@pipeline-builder` (pégalo tal cual y captura la pantalla):

```
@pipeline-builder Build a new use case UCDemo from the dataset UCDemo/data/datos.csv
(semicolon separated). Inputs I1..I7, outputs O1 and O2, requirement Q90 < 0.10 per
output. Choose champion and baseline models with justification, scaffold the full
SF v2.2 pipeline, do a smoke run, then the production run, and finish with the
verdict table and corrective actions.
```

3. Capturas para las diapositivas (mismo formato que los ejemplos 04-06 del PDF):
   - el PLAN que propone el agente (equivalente a la slide 15 del PDF),
   - el scaffolding: árbol UCDemo/ + un YAML generado (SF_6 con la justificación del modelo en comentarios),
   - la ejecución: "Finished with N steps" + terminal del run_pipeline.py,
   - el cierre: tabla veredicto Q90 PASS/FAIL + acción correctiva, y el executive summary generado.
4. Cierre del ejemplo: el agente no solo escribe código, ejecuta el pipeline y lee su propia validación. La skill garantiza que respete las convenciones (job_name, seeds, catálogo, test set intocable).

Puntos que encajan con el roadmap del PDF: es un "test case" nuevo para la lista de la slide 13 (implementation of a simple tool / refactoring / tests), usa un solo agente + skill (nivel 2 de los building blocks) y deja preparada la versión multi-agente.

## Más ideas de agentes para SF (por orden de esfuerzo)

1. Validation interpreter (hecho): pegar la ruta de un UC y obtener veredicto + acciones. Cero riesgo, valor inmediato.
2. Requirement fixer: detecta outputs que cruzan cero y propone el criterio range-normalised en SF_1 (el caso UCAirfoils).
3. Data-enrichment planner: lee el bias heatmap y los puntos isolated del VTP y redacta la petición de nuevas simulaciones (regímenes concretos del envelope).
4. UC migrator: convierte un script/notebook legacy de otra disciplina al esqueleto SF (el caso UCCpHTP cuando llegue el dataset CFD).
5. Hyperparameter runner: solo toca SF_6.yaml, relanza y compara runs en MLflow; nunca toca código.
6. Trio coder/reviewer/tester sobre validationlib (mismo patrón que la slide 34 del PDF), p. ej. para arreglar los rough edges conocidos (zScoreBias en el pipeline paramétrico, waldwolfo con NumPy 2).
7. Report-to-slides: regenerar las diapositivas de resultados del deck a partir del último executive summary (lo que hemos hecho a mano esta semana, automatizado).
8. Multi-agente (fase H100 del roadmap): orchestrator (Qwen 72B) reparte a builder/validator/tester; ACP entre ellos.

## Riesgos/límites a mencionar (de vuestra propia experiencia en el PDF)

- Rate limit 30 peticiones/min y errores 502: el agente debe hacer runs por lotes y reintentar con backoff.
- Modelos pequeños (Gemma 26B) hacen interfaces "de primera iteración" (slide 28): la skill compensa fijando convenciones; para scaffolding completo usar GPT 120B o el cluster H100.
- Guardarraíles del builder: no toca src/ ni validationlib/, para tras 2 fallos idénticos, seeds fijas.
