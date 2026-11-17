"""Run from the repository root: python -m shiny run apps/shiny/app.py"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import math
import textwrap
from urllib.parse import parse_qs

import pandas as pd
from matplotlib.figure import Figure
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from shiny import App, reactive, render, ui

from apps.shiny.core import (
    AGGREGATIONS,
    AGGREGATION_FORMULAS,
    MEASURES,
    PARAMETERS,
    calculate_many,
    calculate_many_per_class,
    detect_csv_has_header,
    estimate_workload,
    format_method_label,
    format_method_latex,
    grid_view,
    header_weight_vector,
    numbers,
    validate_csv_upload,
)
from apps.shiny.evaluation import CRITERIA, METRIC_DOCS, evaluate, evaluate_proba, text_decisions

EXAMPLE = Path(__file__).with_name('example.csv')
EXAMPLE_BINARY = Path(__file__).with_name('example_binary.csv')
EXAMPLE_MULTICLASS = Path(__file__).with_name('example_multiclass.csv')
LOGO_PATH = Path(__file__).with_name('means-icon.svg')
LOGO_SVG = LOGO_PATH.read_text(encoding='utf-8') if LOGO_PATH.is_file() else ''


def _package_version():
    try:
        return package_version('aggregationslib')
    except PackageNotFoundError:
        return ''


APP_VERSION = _package_version()
CODE_FORMATTER = HtmlFormatter(cssclass='python-code', style='friendly')
MAX_WORKERS = 2
MAX_QUEUED_TASKS = 6
COMPUTATION_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='agg-worker')
_COMPUTATION_SEMAPHORE = None
_QUEUED_TASKS = 0
_ACTIVE_TASKS = 0


def get_computation_semaphore():
    global _COMPUTATION_SEMAPHORE
    if _COMPUTATION_SEMAPHORE is None:
        _COMPUTATION_SEMAPHORE = asyncio.Semaphore(MAX_WORKERS)
    return _COMPUTATION_SEMAPHORE


def view_picker(id, label):
    return ui.div(ui.input_radio_buttons(id, label,
                  {'short': '5 … 5', 'all': 'All rows'}, selected='short', inline=True),
                  class_='view-picker', title='Compact view shows the first and last five rows. Full view shows all rows.')


app_ui = ui.page_fluid(
    ui.div(
    ui.div(
        ui.div(
            ui.div(ui.HTML(LOGO_SVG), class_='app-logo') if LOGO_SVG else None,
            ui.h2('aggregationslib', class_='mb-0'),
            ui.span(f'v{APP_VERSION}', class_='badge text-bg-light border text-secondary') if APP_VERSION else None,
            class_='d-flex align-items-center gap-2',
        ),
        ui.div(
            ui.tags.a('Documentation', href='https://wgalka.github.io/Means/',
                      target='_blank', rel='noopener noreferrer',
                      class_='link-secondary link-offset-2 link-underline-opacity-0 link-underline-opacity-75-hover small'),
            ui.tags.a('PyPI', href='https://pypi.org/project/aggregationslib/',
                      target='_blank', rel='noopener noreferrer',
                      class_='link-secondary link-offset-2 link-underline-opacity-0 link-underline-opacity-75-hover small'),
            class_='d-flex align-items-center gap-3',
        ),
        class_='d-flex flex-wrap align-items-center justify-content-between gap-2',
    ),
    ui.p('Upload your data, choose an aggregation method, and download the results. Add ground truth labels to evaluate classification.', class_='text-muted'),
    ui.tags.style('.container-fluid {max-width: 1600px;} pre {max-height: 500px; overflow: auto;} '
                  '.app-logo svg {width: 44px; height: 44px; display: block;} '
                  '.app-logo path[fill="#0b2a4f"] {fill: #0b2a4f;} '
                  '.app-logo path[fill="#256487"] {fill: #256487;} '
                  '.app-logo path[fill="#36a08f"] {fill: #36a08f;} '
                  '.app-logo path[fill="#3e94cc"] {fill: #3e94cc;} '
                  '.app-logo path[fill="#7964a2"] {fill: #7964a2;} '
                  '#preview th[scope="col"] > div {display: inline;}'),
    ui.tags.style('#preview th .aggregation-weight {display: block; margin-bottom: .5rem;} '
                  '.aggregation-weight input {width: 100%; min-width: 6rem; font-weight: normal;} '
                  '.aggregation-weight span {display: block; font-size: .75rem; font-weight: normal;}'),
    ui.tags.style('''
        .view-picker {display: flex; justify-content: flex-end; margin-bottom: .4rem;}
        .view-picker .shiny-input-container {display: flex; align-items: center; gap: .6rem; width: auto; margin: 0;}
        .view-picker .control-label {margin: 0; font-size: .8rem; color: var(--bs-secondary-color); font-weight: normal;}
        .view-picker .shiny-options-group {display: flex; border: 1px solid var(--bs-border-color); border-radius: .5rem; overflow: hidden;}
        .view-picker .radio-inline {position: relative; margin: 0; padding: .3rem .7rem; cursor: pointer; font-size: .8rem;}
        .view-picker .radio-inline + .radio-inline {border-left: 1px solid var(--bs-border-color);}
        .view-picker input {position: absolute; opacity: 0; width: 1px; height: 1px;}
        .view-picker .radio-inline:has(input:checked) {background: var(--bs-primary); color: white;}
        .view-picker .radio-inline:focus-within {outline: 2px solid var(--bs-primary); outline-offset: -2px;}
    '''),
    ui.tags.style(CODE_FORMATTER.get_style_defs('.python-code') +
                  '\n.python-code {padding: 0; background: transparent !important; border: 0;} '
                  '.python-code pre {margin: 0; padding: 1rem; border-radius: 6px; '
                  'background: #f8f9fa; border: 1px solid var(--bs-border-color, #dee2e6); '
                  'white-space: pre; max-height: 500px; overflow: auto;}'),
    ui.tags.script(ui.HTML(Path(__file__).with_name('column-selection.js').read_text(encoding='utf-8'))),
    ui.head_content(
        ui.tags.link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'),
        ui.tags.script(src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'),
        ui.tags.script(src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js'),
    ),
    ui.tags.script(ui.HTML('''
        (function () {
            function renderLatex(root) {
                if (!window.renderMathInElement || !root) return;
                window.renderMathInElement(root, {
                    delimiters: [{left: "\\\\(", right: "\\\\)", display: false}],
                    throwOnError: false
                });
            }
            function renderTargets() {
                renderLatex(document.getElementById("method_list"));
                renderLatex(document.getElementById("selected_formula"));
            }
            window.addEventListener("load", renderTargets);
            document.addEventListener("DOMContentLoaded", function () {
                renderTargets();
                if (window.jQuery) {
                    window.jQuery(document).on("shiny:value", function (e) {
                        if (e && (e.name === "method_list" || e.name === "selected_formula")) {
                            requestAnimationFrame(function () { renderLatex(e.target); });
                        }
                    });
                }
                ["method_list", "selected_formula"].forEach(function (id) {
                    var el = document.getElementById(id);
                    if (!el || !window.MutationObserver) return;
                    new MutationObserver(function () {
                        renderLatex(el);
                    }).observe(el, {childList: true, subtree: true});
                });
            });
            if (window.jQuery) {
                window.Shiny.addCustomMessageHandler("eval-mode", function (msg) {
                    document.body.classList.toggle("is-multiclass", !!(msg && msg.multiclass));
                });
                window.Shiny.addCustomMessageHandler("save-config", function (msg) {
                    try {
                        localStorage.setItem("aggregationslib.method_queue.v1", JSON.stringify((msg && msg.methods) || []));
                    } catch (err) {}
                });
                window.jQuery(document).on("shiny:sessioninitialized", function () {
                    var saved = null;
                    try {
                        var raw = localStorage.getItem("aggregationslib.method_queue.v1");
                        if (raw !== null) saved = JSON.parse(raw);
                    } catch (err) { saved = null; }
                    window.Shiny.setInputValue("restored_config", {methods: Array.isArray(saved) ? saved : null, nonce: Date.now()}, {priority: "event"});
                });
            }
        })();
    ''')),

    ui.tags.style('''
        body {background: #f5f7fa;}
        .app-shell {max-width: 1120px; margin: 2rem auto; padding: 0 1rem;
                    display: flex; flex-direction: column; gap: 1.25rem;}
        .app-shell > h2, .app-shell > p {margin-bottom: 0;}
        .app-shell .card {box-shadow: 0 2px 8px #18263c08;}
        .app-shell .card-header {font-weight: 600; padding: 1rem 1.25rem;}
        .app-shell .shiny-input-container {max-width: 100%;}
        .app-shell .input-group {
            display: flex;
            align-items: stretch;
            width: 100%;
            border: var(--bs-border-width, 1px) solid #8D959E;
            border-radius: var(--bs-border-radius, .375rem);
            overflow: hidden;
            background-color: var(--bs-body-bg, #fff);
            transition: border-color .15s ease-in-out, box-shadow .15s ease-in-out;
        }
        .app-shell .input-group:focus-within {
            border-color: #80bde1;
            outline: 0;
            box-shadow: 0 0 0 .25rem rgba(0, 123, 194, .25);
        }
        .app-shell .input-group .input-group-btn {display: flex; align-items: stretch; margin: 0;}
        .app-shell .input-group .btn-file {
            display: flex;
            align-items: center;
            height: 100%;
            background-color: var(--bs-tertiary-bg, #f8f9fa);
            border: 0;
            border-right: var(--bs-border-width, 1px) solid #8D959E;
            border-radius: 0;
            color: var(--bs-body-color, #212529);
            font-size: .9375rem;
            padding: .375rem .75rem;
            line-height: 1.5;
            box-shadow: none;
        }
        .app-shell .input-group .btn-file:hover,
        .app-shell .input-group .btn-file:active,
        .app-shell .input-group .btn-file:focus {
            background-color: var(--bs-secondary-bg, #e9ecef);
            box-shadow: none;
            outline: 0;
        }
        .app-shell .input-group .form-control {
            flex: 1 1 auto;
            width: 1%;
            border: 0;
            border-radius: 0;
            font-size: .9375rem;
            background-color: var(--bs-body-bg, #fff);
            box-shadow: none;
        }
        .app-shell .input-group .form-control:focus {box-shadow: none; outline: 0;}
        .app-shell .shiny-file-input-progress {height: 4px; margin-top: .35rem; margin-bottom: 0;}
        .app-shell .list-group-item.border-primary {border-top-width: var(--bs-border-width, 1px); z-index: 1;}
        .aggregation-controls {display: flex; flex-wrap: wrap; align-items: flex-end; gap: .75rem; width: 100%;}
        .aggregation-controls > .aggregation-main {flex: 2 1 260px; min-width: 0;}
        .aggregation-controls > .shiny-panel-conditional {flex: 1 1 160px; min-width: 0;}
        @media (min-width: 768px) {
            .aggregation-controls {flex-wrap: nowrap;}
        }
        .aggregation-controls .shiny-input-container {width: 100%; margin-bottom: 0;}
        .run-bar {
            position: sticky;
            bottom: .75rem;
            z-index: 1030;
            width: fit-content;
            max-width: 100%;
            margin-left: auto;
            background: rgba(255, 255, 255, .96);
            backdrop-filter: blur(6px);
            border: 1px solid var(--bs-border-color, #dee2e6);
            border-radius: .6rem;
            padding: .4rem .6rem;
            box-shadow: 0 4px 14px rgba(24, 38, 60, .1);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: .5rem;
        }
        .run-bar > .run-status {flex: 0 1 auto; min-width: 0;}
        .run-bar .alert {padding: .25rem .6rem; font-size: .825rem;}
        body.is-multiclass .binary-only {display: none !important;}
    '''),
    ui.card(ui.card_header('1. Input data'),
        ui.input_radio_buttons('input_mode', 'Input mode',
            {'single': 'Single CSV', 'per_class': 'One CSV per class (predict_proba fusion)'},
            selected='single', inline=True),
        ui.layout_columns(
            ui.div(
                ui.panel_conditional("input.input_mode === 'single'",
                    ui.input_file('file', 'Upload a CSV file', accept=['.csv'], width='100%'),
                    ui.help_text('Each row contains one vector to aggregate.')),
                ui.panel_conditional("input.input_mode === 'per_class'",
                    ui.input_numeric('num_classes', 'Number of classes', 2, min=2, max=20, width='100%'),
                    ui.output_ui('per_class_uploads'),
                    ui.help_text('Upload one CSV per class containing model probability columns (same row order).')),
            ),
            ui.div(ui.input_select('separator', 'CSV delimiter', {',': 'Comma', ';': 'Semicolon', '\t': 'Tab'}, width='100%'),
                   ui.input_select('decimal', 'Decimal separator', {'.': 'Period', ',': 'Comma'}, width='100%'),
                   ui.input_checkbox('header', 'First row contains column headers', True)))),
    ui.card(ui.card_header('2. Ground truth labels (optional)'),
            ui.input_radio_buttons('decision_source', 'Ground truth labels',
                {'none': 'No labels', 'input_column': 'Column from input CSV', 'text': 'Paste labels', 'csv': 'Upload separate CSV'}, inline=True),
            ui.panel_conditional("input.decision_source === 'input_column'",
                ui.input_select('input_decision_column', 'Decision column', [])),
            ui.panel_conditional("input.decision_source === 'text'",
                ui.input_text_area('decision_text', 'Label vector', width='100%', placeholder='0 1 0 1 …')),
            ui.panel_conditional("input.decision_source === 'csv'",
                ui.input_file('decision_file', 'CSV with labels', accept=['.csv'], width='100%'),
                ui.input_select('decision_separator', 'Label CSV delimiter', {',': 'Comma', ';': 'Semicolon', '\t': 'Tab'}),
                ui.input_checkbox('decision_header', 'Label CSV has a header row', True),
                ui.input_select('decision_column', 'Label column', [])),

        ui.panel_conditional("input.decision_source !== 'none'",
            ui.help_text('Enter one label per row in the original order.'),
            ui.div(
                ui.layout_columns(
                    ui.input_text('positive_label', 'Positive class label', '1'),
                    ui.input_select('score_direction', 'Predict positive when the score is', {'higher': 'At or above the threshold', 'lower': 'At or below the threshold'})),
                class_='binary-only',
            ))),
    ui.card(ui.card_header('3. Aggregation and parameters'),
        ui.output_ui('method_list'),
        ui.div(
            ui.div(
                ui.input_select('aggregation', 'Select aggregation to add / configure', AGGREGATIONS, selected='A_ch', width='100%'),
                class_='aggregation-main',
            ),
            ui.panel_conditional("['A_pw','A_ex','A_lm','A_lm1'].includes(input.aggregation)",
                                 ui.input_numeric('r', 'Parameter r', 2, width='100%')),
            ui.panel_conditional("['A_ex2','A_ex3','A_amn','A_amx','A_oln'].includes(input.aggregation)",
                                 ui.input_numeric('p', 'Parameter p', 1, width='100%')),
            ui.panel_conditional("['A_ex2','A_ex3'].includes(input.aggregation)",
                                 ui.input_numeric('q', 'Parameter q', 2, width='100%')),
            ui.panel_conditional("['A_ch','A_sg'].includes(input.aggregation)",
                ui.input_select('measure', 'Measure', MEASURES, width='100%')),
            ui.panel_conditional("['A_ch','A_sg'].includes(input.aggregation) && input.measure === 'exponential_cardinality'",
                ui.input_numeric('measure_q', 'Measure parameter q > 0', 1, min=0, step=.1, width='100%')),
            class_='aggregation-controls',
        ),
        ui.panel_conditional("['A_ch','A_sg'].includes(input.aggregation) && ['weighted','owa','k_fuzzy'].includes(input.measure)",
            ui.div(
                ui.input_text('measure_weights', 'Weights (space or comma separated)', placeholder='e.g. 0.5 0.3 0.2', width='100%'),
                ui.output_ui('measure_weights_hint'),
                class_='mt-2',
            ),
        ),
        ui.output_ui('selected_formula'),
        ui.panel_conditional("['A_ch','A_sg'].includes(input.aggregation)",
            ui.help_text('The integrals accept values in [0, 1]. With the uniform measure, the Choquet integral equals the arithmetic mean. The first OWA weight applies to the largest value.')),
        ui.div(
            ui.input_action_button('add_method', '+ Add with current parameters', class_='btn-sm btn-outline-primary'),
            ui.input_action_button('clear_methods', 'Clear list', class_='btn-sm btn-outline-secondary'),
            ui.input_action_button('reset_methods', 'Reset defaults', class_='btn-sm btn-outline-secondary'),
            class_='d-flex flex-wrap gap-2 mt-2',
        ),
    ),
    ui.panel_conditional("input.input_mode === 'single'",
        ui.card(ui.card_header('4. Columns'),
            ui.div(ui.input_action_button('select_all', 'Select all', class_='btn-sm btn-outline-primary'),
                   ui.input_action_button('select_none', 'Deselect all', class_='btn-sm btn-outline-secondary'),
                   class_='d-flex gap-2'),
            ui.help_text('The default view shows the first and last five rows. Use the header checkboxes to select columns for aggregation. Click a column name to sort the full table. The CSV export keeps the original row order.'),
            view_picker('preview_view', 'Input view'), ui.output_data_frame('preview'))),

    ui.div(
        ui.input_task_button(
            'run_calc',
            ui.span(
                ui.HTML('<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11l9-5.5-9-5.5z"/></svg>'),
                'Run aggregation',
                class_='d-inline-flex align-items-center gap-2',
            ),
            label_busy='Calculating…',
            class_='btn-sm btn-primary',
        ),
        ui.div(ui.output_ui('status'), class_='run-status'),
        class_='run-bar',
    ),
    ui.card(ui.card_header('5. Aggregation results'),
            ui.help_text('The default view shows the first and last five rows. Sorting applies to all results.'),
            view_picker('results_view', 'Results view'),
            ui.output_data_frame('results'), ui.output_ui('download_ui')),

    ui.panel_conditional("input.decision_source !== 'none'",
        ui.card(ui.card_header('6. Classification evaluation and threshold'),
            ui.div(
                ui.input_select('threshold_criterion', 'Threshold criterion', CRITERIA),
                ui.input_radio_buttons('threshold_mode', 'Threshold t',
                    {'auto': 'Automatic', 'manual': 'Manual'}, selected='auto', inline=True),
                ui.panel_conditional("input.threshold_mode === 'manual'",
                    ui.input_numeric('manual_threshold', 'Custom threshold t', .5, step=.01)),
                ui.input_action_button('reset_threshold', 'Reset to Youden threshold', class_='btn-sm btn-outline-secondary'),
                class_='binary-only',
            ),
                ui.output_ui('eval_method_ui'),
                ui.output_ui('evaluation_status'),
                ui.output_ui('roc_plot_ui'),
                ui.output_ui('metric_table'),
                ui.output_ui('confusion_plots_ui'),
                ui.help_text('This report selects and evaluates the threshold on the same data. To assess generalization, choose a threshold on validation data and evaluate it on a separate test set. Metrics with a zero denominator are set to 0.'),
                ui.output_ui('classification_download_ui'))),
    ui.card(ui.card_header('Run this in Python'), ui.p('If you uploaded a file, save a local copy as input.csv.'),
            ui.output_ui('code')),

    class_='app-shell'),
    title='aggregationslib · Table aggregation',
)



def _default_weights_for(n):
    if n <= 0:
        return ''
    if n == 3:
        return '0.5 0.3 0.2'
    base = round(1 / n, 4)
    vals = [base] * (n - 1)
    vals.append(round(1 - sum(vals), 4))
    return ' '.join(f'{v:g}' for v in vals)


def _default_methods():
    methods = []
    for name in ('A_ch', 'A_sg'):
        for measure in ('uniform', 'relative', 'exponential_cardinality', 'weighted', 'owa'):
            params = {'measure_q': 0.5} if measure == 'exponential_cardinality' else {}
            weights = '0.5 0.3 0.2' if measure in ('weighted', 'owa') else ''
            methods.append({
                'name': name,
                'params': params,
                'measure': measure,
                'weights': weights,
                'auto_weights': measure in ('weighted', 'owa'),
            })
    return methods


def server(input, output, session):
    selected_columns = reactive.value([])
    weight_values = reactive.value({})
    sort_states = reactive.value({})
    method_queue = reactive.value(_default_methods())
    editing_index = reactive.value(0)
    suppress_auto_add = reactive.value(False)
    config_ready = reactive.value(False)
    data_version = reactive.value(0)
    last_completed_sig = reactive.value(None)
    in_flight_sig = reactive.value(None)
    initial_triggered = reactive.value(False)
    progress_state = {'phase': 'idle', 'done': 0, 'total': 0}

    def _current_method_item():
        name = input.aggregation()
        params = {key: getattr(input, key)() for key in PARAMETERS.get(name, ())}
        measure = input.measure() if name in ('A_ch', 'A_sg') else 'uniform'
        if name in ('A_ch', 'A_sg') and measure == 'exponential_cardinality':
            params['measure_q'] = input.measure_q()
        weights = input.measure_weights() if (name in ('A_ch', 'A_sg') and measure in ('weighted', 'owa', 'k_fuzzy')) else ''
        return {'name': name, 'params': params, 'measure': measure, 'weights': weights}

    @render.ui
    def measure_weights_hint():
        cols = [c for c in data()[0].columns if c in selected_columns()]
        n = len(cols)
        measure = input.measure()
        if measure == 'owa':
            text = f'Enter {n} weights in [0, 1] by descending value rank (largest first). Weights must sum to 1.'
        elif measure == 'k_fuzzy':
            text = f"Enter {n} singleton weights in [0, 1] for columns ({', '.join(cols)}). They may sum to a value other than 1."
        else:
            text = f"Enter {n} weights in [0, 1] for columns ({', '.join(cols)}). Weights must sum to 1."
        return ui.div(text, class_='form-text')

    @render.ui
    def selected_formula():
        item = _current_method_item()
        name = item['name']
        latex = format_method_latex(name, item.get('measure', 'uniform'), item.get('params'), item.get('weights', ''))
        if not latex:
            return None
        return ui.p(
            ui.tags.code(name), ' — ', AGGREGATIONS.get(name, name), ': ',
            ui.tags.span(latex),
            class_='text-muted small mb-2 mt-1',
        )

    @render.ui
    def method_list():
        items = method_queue()
        active_idx = editing_index()
        if not items:
            return ui.div(
                'No aggregations in the list yet. Select a method below to add it.',
                class_='alert alert-light border text-muted small py-2 mb-3',
            )
        rows = []
        for idx, item in enumerate(items):
            name = item['name']
            is_active = active_idx == idx
            label = format_method_label(item)
            formula = format_method_latex(name, item.get('measure', 'uniform'), item.get('params'), item.get('weights', ''))
            title = AGGREGATIONS.get(name, name)
            def _action_btn(text, op, title_attr, btn_cls='btn-outline-secondary', disabled=False):
                attrs = {
                    'type': 'button',
                    'class_': f'btn btn-sm {btn_cls}',
                    'title': title_attr,
                    'onclick': f"Shiny.setInputValue('method_action', {{op: '{op}', index: {idx}, nonce: Date.now()}})",
                }
                if disabled:
                    attrs['disabled'] = 'disabled'
                return ui.tags.button(text, **attrs)

            rows.append(
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span(f'{idx + 1}.', class_='text-muted small me-1'),
                            ui.tags.code(label, class_='fw-semibold'),
                            ui.span(f' · {title}', class_='small ms-1'),
                            ui.span('Editing', class_='badge text-bg-primary ms-2') if is_active else None,
                        ),
                        ui.div(formula, class_='text-muted small mt-1') if formula else None,
                        class_='flex-grow-1',
                    ),
                    ui.div(
                        _action_btn('↑', 'up', 'Move up', disabled=idx == 0),
                        _action_btn('↓', 'down', 'Move down', disabled=idx == len(items) - 1),
                        _action_btn('Edit', 'edit', 'Load into editor below',
                                    btn_cls='btn-primary' if is_active else 'btn-outline-primary'),
                        _action_btn('×', 'remove', 'Remove', btn_cls='btn-outline-danger'),
                        class_='d-flex align-items-center gap-1',
                    ),
                    class_=f"list-group-item d-flex flex-wrap align-items-center justify-content-between gap-2 {'border-primary bg-light' if is_active else ''}",
                )
            )
        return ui.div(
            ui.div(f'Selected aggregations ({len(items)}):', class_='small fw-semibold text-muted mb-1'),
            ui.div(*rows, class_='list-group mb-3'),
        )

    @reactive.effect
    @reactive.event(input.aggregation)
    def on_aggregation_select():
        if suppress_auto_add():
            suppress_auto_add.set(False)
            return
        item = _current_method_item()
        with reactive.isolate():
            items = list(method_queue())
            idx = editing_index()
            if idx is not None and 0 <= idx < len(items) and items[idx]['name'] == item['name']:
                return
            items.append(item)
            method_queue.set(items)
            editing_index.set(len(items) - 1)

    @reactive.effect
    @reactive.event(input.r, input.p, input.q, input.measure, input.measure_q, input.measure_weights)
    def sync_active_method_params():
        item = _current_method_item()
        with reactive.isolate():
            idx = editing_index()
            items = list(method_queue())
            if idx is not None and 0 <= idx < len(items) and items[idx]['name'] == item['name']:
                if items[idx] != item:
                    items[idx] = item
                    method_queue.set(items)

    @reactive.effect
    @reactive.event(input.add_method)
    def add_method_to_queue():
        item = _current_method_item()
        with reactive.isolate():
            items = list(method_queue())
            items.append(item)
            method_queue.set(items)
            editing_index.set(len(items) - 1)

    @reactive.effect
    @reactive.event(input.clear_methods)
    def clear_method_queue():
        method_queue.set([])
        editing_index.set(None)

    @reactive.effect
    @reactive.event(input.reset_methods)
    def reset_method_queue():
        defaults = _default_methods()
        n = len(selected_columns())
        if 0 < n <= 12:
            for item in defaults:
                if item.get('auto_weights'):
                    item['weights'] = _default_weights_for(n)
        method_queue.set(defaults)
        editing_index.set(0)

    @reactive.effect
    @reactive.event(input.restored_config)
    def restore_saved_config():
        payload = input.restored_config()
        saved = payload.get('methods') if isinstance(payload, dict) else None
        if isinstance(saved, list):
            valid = []
            for item in saved:
                if not isinstance(item, dict) or item.get('name') not in AGGREGATIONS:
                    continue
                valid.append({
                    'name': item['name'],
                    'params': item.get('params') if isinstance(item.get('params'), dict) else {},
                    'measure': item.get('measure', 'uniform') if item.get('measure', 'uniform') in MEASURES else 'uniform',
                    'weights': str(item.get('weights', '')),
                    'auto_weights': bool(item.get('auto_weights', False)),
                })
            method_queue.set(valid)
            editing_index.set(0 if valid else None)
        config_ready.set(True)

    @reactive.effect
    async def persist_saved_config():
        items = method_queue()
        if config_ready():
            await session.send_custom_message('save-config', {'methods': items})

    @reactive.effect
    @reactive.event(input.method_action)
    def handle_method_action():
        action = input.method_action()
        if not isinstance(action, dict):
            return
        op = action.get('op')
        idx = action.get('index')
        with reactive.isolate():
            items = list(method_queue())
            active = editing_index()
        if not isinstance(idx, int) or not 0 <= idx < len(items):
            return
        if op == 'up' and idx > 0:
            items[idx - 1], items[idx] = items[idx], items[idx - 1]
            method_queue.set(items)
            if active == idx:
                editing_index.set(idx - 1)
            elif active == idx - 1:
                editing_index.set(idx)
        elif op == 'down' and idx < len(items) - 1:
            items[idx + 1], items[idx] = items[idx], items[idx + 1]
            method_queue.set(items)
            if active == idx:
                editing_index.set(idx + 1)
            elif active == idx + 1:
                editing_index.set(idx)
        elif op == 'remove':
            items.pop(idx)
            method_queue.set(items)
            if active == idx:
                editing_index.set(min(idx, len(items) - 1) if items else None)
            elif active is not None and active > idx:
                editing_index.set(active - 1)
        elif op == 'edit':
            item = items[idx]
            editing_index.set(idx)
            if item['name'] != input.aggregation():
                suppress_auto_add.set(True)
                ui.update_select('aggregation', selected=item['name'])
            params = item.get('params', {})
            for key in ('r', 'p', 'q'):
                if key in params:
                    ui.update_numeric(key, value=params[key])
            if item['name'] in ('A_ch', 'A_sg'):
                ui.update_select('measure', selected=item.get('measure', 'uniform'))
                if 'measure_q' in params:
                    ui.update_numeric('measure_q', value=params['measure_q'])
                ui.update_text('measure_weights', value=item.get('weights', ''))

    @reactive.calc
    def test_preset():
        query = parse_qs(session.clientdata.url_search().lstrip('?'))
        test_value = (query.get('test') or [None])[0]
        if test_value in ('1', 'binary'):
            return 'binary'
        if test_value in ('2', 'multiclass'):
            return 'multiclass'
        if query.get('tester') == ['1']:
            return 'tester'
        return None

    @reactive.effect
    def load_test_decisions():
        preset = test_preset()
        if preset == 'tester':
            frame = pd.read_csv(EXAMPLE)
            decisions = (frame.mean(axis=1) >= .5).astype(int)
            ui.update_text_area('decision_text', value=' '.join(map(str, decisions)))
            ui.update_radio_buttons('decision_source', selected='text')
        elif preset in ('binary', 'multiclass'):
            ui.update_radio_buttons('decision_source', selected='input_column')
            ui.update_select('input_decision_column', selected='label')

    @reactive.effect
    @reactive.event(input.file, input.separator, input.decimal)
    def auto_detect_header():
        file = input.file()
        if file and input.separator() != input.decimal():
            try:
                has_header = detect_csv_has_header(file[0]['datapath'], input.separator(), input.decimal())
                ui.update_checkbox('header', value=has_header)
            except OSError:
                pass

    @render.ui
    def per_class_uploads():
        count = input.num_classes()
        k = max(2, min(20, int(count))) if count is not None else 2
        rows = []
        for idx in range(k):
            rows.append(
                ui.div(
                    ui.input_text(f'class_name_{idx}', f'Class {idx + 1} label', value=str(idx), width='100%'),
                    ui.input_file(f'class_file_{idx}', f'CSV for class {idx + 1}', accept=['.csv'], width='100%'),
                    class_='d-grid gap-2 mb-2',
                    style='grid-template-columns: minmax(110px, 1fr) minmax(200px, 2fr); align-items: end;',
                )
            )
        return ui.div(*rows)

    @reactive.calc
    def per_class_data():
        if input.input_mode() != 'per_class':
            return [], [], None
        try:
            if input.separator() == input.decimal():
                raise ValueError('The CSV delimiter and decimal separator must be different.')
            count = input.num_classes()
            k = max(2, min(20, int(count))) if count is not None else 2
            frames = []
            names = []
            for idx in range(k):
                cname = (getattr(input, f'class_name_{idx}')() or '').strip()
                if not cname:
                    raise ValueError(f'Enter a label for class {idx + 1}.')
                files = getattr(input, f'class_file_{idx}')()
                if not files:
                    raise ValueError(f'Upload a CSV file for class {cname!r}.')
                validate_csv_upload(files[0]['datapath'])
                df = pd.read_csv(files[0]['datapath'], sep=input.separator(),
                                 decimal=input.decimal(), header=0 if input.header() else None)
                if not input.header():
                    df.columns = [f'x{i + 1}' for i in range(len(df.columns))]
                frames.append(df)
                names.append(cname)
            return frames, names, None
        except (ValueError, OSError, UnicodeError) as error:
            return [], [], str(error)

    @reactive.calc
    def data():
        try:
            file = input.file()
            if file:
                if input.separator() == input.decimal():
                    raise ValueError('The CSV delimiter and decimal separator must be different.')
                validate_csv_upload(file[0]['datapath'])
                df = pd.read_csv(file[0]['datapath'], sep=input.separator(),
                                 decimal=input.decimal(), header=0 if input.header() else None)
                if not input.header():
                    df.columns = [f'x{i + 1}' for i in range(len(df.columns))]
                return df, None
            preset = test_preset()
            if preset == 'tester':
                return pd.read_csv(EXAMPLE), None
            if preset == 'binary':
                return pd.read_csv(EXAMPLE_BINARY), None
            if preset == 'multiclass':
                return pd.read_csv(EXAMPLE_MULTICLASS), None
            return pd.DataFrame(), 'Upload a CSV file to get started.'
        except (ValueError, OSError, UnicodeError) as error:
            return pd.DataFrame(), str(error)

    @reactive.effect
    def update_columns():
        df, error = data()
        if not input.file() and test_preset() in ('binary', 'multiclass'):
            selected_columns.set([column for column in df.columns if column != 'label'])
        else:
            selected_columns.set(list(df.columns))
        weight_values.set({})
        sort_states.set({})

    @reactive.calc
    def weight_kind():
        return None

    @reactive.effect
    async def sync_headers():
        await session.send_custom_message('aggregation-columns', {
            'columns': list(data()[0].columns), 'selected': selected_columns(),
            'weightKind': weight_kind(),
            'weights': weight_values().get(weight_kind(), {}),
            'sortStates': sort_states(),
        })

    @reactive.effect
    @reactive.event(input.table_sort)
    def sort_table():
        message = input.table_sort()
        if message['grid'] in ('preview', 'results') and message['direction'] in ('asc', 'desc'):
            states = dict(sort_states())
            states[message['grid']] = {'index': int(message['index']), 'direction': message['direction']}
            sort_states.set(states)

    @reactive.effect
    @reactive.event(input.header_weights)
    def weight_edit():
        message = input.header_weights()
        if message['kind'] == weight_kind():
            values = dict(weight_values())
            values[message['kind']] = message['values']
            weight_values.set(values)

    @render.ui
    def weight_hint():
        kind = weight_kind()
        if kind:
            text = f'Enter {len(selected_columns())} weights in the fields above the selected column headers.'
            if kind == 'owa':
                text += ' OWA assigns weights by rank within each row, starting with the largest value.'
            elif kind == 'k_fuzzy':
                text += ' These are singleton weights and may sum to a value other than 1.'
            if kind in ('weighted', 'owa'):
                text += ' The weights must sum to 1.'
            return ui.p(text, class_='text-muted')

    @reactive.effect
    @reactive.event(input.header_columns)
    def header_selection():
        selected_columns.set([column for column in data()[0].columns
                              if column in input.header_columns()])

    @reactive.effect
    @reactive.event(input.select_all)
    def select_all():
        selected_columns.set(list(data()[0].columns))

    @reactive.effect
    @reactive.event(input.select_none)
    def select_none():
        selected_columns.set([])

    @reactive.effect
    def _sync_default_weights_length():
        cols = selected_columns()
        n = len(cols)
        if n <= 0 or n > 12:
            return
        with reactive.isolate():
            items = list(method_queue())
            changed = False
            for idx, item in enumerate(items):
                if item.get('auto_weights'):
                    try:
                        cur_len = len(numbers(item.get('weights', '')))
                    except ValueError:
                        cur_len = 0
                    if cur_len != n:
                        updated = dict(item)
                        updated['weights'] = _default_weights_for(n)
                        items[idx] = updated
                        changed = True
            if changed:
                method_queue.set(items)
                active = editing_index()
                if active is not None and 0 <= active < len(items) and items[active].get('auto_weights'):
                    ui.update_text('measure_weights', value=items[active]['weights'])
    @reactive.effect
    def _bump_data_version():
        data()
        with reactive.isolate():
            data_version.set(data_version() + 1)

    def current_request():
        mode = input.input_mode()
        items = method_queue()
        methods_payload = []
        sig_methods = []
        for item in items:
            m_name = item['name']
            m_params = dict(item.get('params', {}))
            m_measure = item.get('measure', 'uniform')
            m_weights = item.get('weights', '') if (m_name in ('A_ch', 'A_sg') and m_measure in ('weighted', 'owa', 'k_fuzzy')) else ''
            methods_payload.append({
                'name': m_name,
                'params': m_params,
                'measure': m_measure,
                'weights': m_weights,
            })
            sig_methods.append((
                m_name,
                tuple(sorted(m_params.items())),
                m_measure,
                m_weights,
            ))

        if mode == 'per_class':
            frames, class_names, error = per_class_data()
            if error:
                return ('per_class_error', error), None, error
            sig = (
                'per_class',
                tuple(class_names),
                tuple((len(f), tuple(f.columns)) for f in frames),
                tuple(sig_methods),
                input.separator(),
                input.decimal(),
                input.header(),
            )
            args = {
                'mode': 'per_class',
                'class_frames': frames,
                'class_names': class_names,
                'methods': methods_payload,
                'separator': input.separator(),
                'decimal': input.decimal(),
                'header': input.header(),
            }
            return sig, args, None

        df, error = data()
        if error:
            return ('error', data_version(), error), None, error
        columns = [column for column in df.columns if column in selected_columns()]
        sig = (
            'single',
            data_version(),
            tuple(columns),
            tuple(sig_methods),
            input.separator(),
            input.decimal(),
            input.header(),
            bool(input.file()),
        )
        args = {
            'mode': 'single',
            'df': df,
            'columns': columns,
            'extra': '',
            'methods': methods_payload,
            'separator': input.separator(),
            'decimal': input.decimal(),
            'header': input.header(),
            'uploaded': bool(input.file()),
        }
        return sig, args, None

    @reactive.extended_task
    async def compute_task(sig, args, data_error, state):
        global _QUEUED_TASKS, _ACTIVE_TASKS
        if data_error:
            return sig, None, '', [], None, data_error
        if not args['methods']:
            return sig, None, '', [], None, 'Add at least one aggregation method.'
        if args['mode'] == 'single' and not args['columns']:
            return sig, None, '', [], None, 'Select at least one column.'
        try:
            total_ram = 0
            total_complexity = 0
            if args['mode'] == 'per_class':
                for m in args['methods']:
                    for df in args['class_frames']:
                        info = estimate_workload(df, list(df.columns), '', m['name'], m['measure'])
                        total_ram += info['estimated_ram_bytes']
                        total_complexity += info['complexity']
            else:
                for m in args['methods']:
                    info = estimate_workload(args['df'], args['columns'], args['extra'], m['name'], m['measure'])
                    total_ram += info['estimated_ram_bytes']
                    total_complexity += info['complexity']
            if total_ram > 128 * 1024 * 1024:
                raise ValueError('Estimated memory usage exceeds the 128 MB server limit.')
            if total_complexity > 2_000_000:
                raise ValueError('The combined list of aggregations is too computationally expensive for interactive processing.')
        except ValueError as err:
            return sig, None, '', [], None, str(err)
        if _QUEUED_TASKS >= MAX_QUEUED_TASKS:
            return sig, None, '', [], None, 'Server is busy (task queue is full). Please wait a moment and try again.'

        def run_sync():
            def on_progress(done, total):
                state['done'] = done
                state['total'] = total

            prepared = [
                {
                    'name': m['name'],
                    'params': m['params'],
                    'measure': m['measure'],
                    'weights': m['weights'],
                }
                for m in args['methods']
            ]
            if args['mode'] == 'per_class':
                return calculate_many_per_class(
                    args['class_frames'], args['class_names'], prepared,
                    args['separator'], args['decimal'], args['header'],
                    on_progress=on_progress,
                )
            table, code, cols = calculate_many(
                args['df'], args['columns'], prepared, args['extra'],
                args['separator'], args['decimal'], args['header'],
                args['uploaded'], on_progress=on_progress,
            )
            return table, code, cols, None

        state['phase'] = 'queued'
        state['done'] = 0
        if args['mode'] == 'per_class':
            state['total'] = sum(len(f) for f in args['class_frames']) * len(args['methods'])
        else:
            state['total'] = len(args['df']) * len(args['methods'])
        _QUEUED_TASKS += 1
        semaphore = get_computation_semaphore()
        try:
            async with semaphore:
                _QUEUED_TASKS -= 1
                _ACTIVE_TASKS += 1
                state['phase'] = 'running'
                try:
                    loop = asyncio.get_running_loop()
                    result, code, result_columns, score_matrices = await loop.run_in_executor(COMPUTATION_EXECUTOR, run_sync)
                    return sig, result, code, result_columns, score_matrices, None
                except (ValueError, TypeError, ArithmeticError, KeyError) as err:
                    return sig, None, '', [], None, str(err)
                finally:
                    _ACTIVE_TASKS -= 1
                    state['phase'] = 'idle'
        except BaseException:
            if state['phase'] == 'queued':
                _QUEUED_TASKS -= 1
            state['phase'] = 'idle'
            raise

    def start_computation(notify_duplicate=False):
        sig, args, data_error = current_request()
        with reactive.isolate():
            if sig == in_flight_sig():
                if notify_duplicate:
                    ui.notification_show('This calculation is already running.', type='warning', duration=3)
                return
            if sig == last_completed_sig():
                if notify_duplicate:
                    ui.notification_show('Results for these exact parameters are already up to date.', type='message', duration=3)
                return
            in_flight_sig.set(sig)
        ui.update_task_button('run_calc', state='busy')
        compute_task(sig, args, data_error, progress_state)

    @reactive.effect
    def auto_run_initial():
        if not initial_triggered() and selected_columns():
            initial_triggered.set(True)
            start_computation(notify_duplicate=False)

    @reactive.effect
    @reactive.event(input.run_calc)
    def trigger_computation():
        start_computation(notify_duplicate=True)

    @reactive.effect
    def sync_task_state():
        status = compute_task.status()
        if status == 'success':
            sig, _, _, _, _, error = compute_task.result()
            with reactive.isolate():
                last_completed_sig.set(sig if error is None else None)
                if in_flight_sig() == sig:
                    in_flight_sig.set(None)
            ui.update_task_button('run_calc', state='ready')
        elif status == 'error':
            with reactive.isolate():
                last_completed_sig.set(None)
                in_flight_sig.set(None)
            ui.update_task_button('run_calc', state='ready')

    @reactive.calc
    def computation():
        _, result, code, _, _, error = compute_task.result()
        return result, code, error

    @reactive.calc
    def computation_columns():
        _, _, _, result_columns, _, _ = compute_task.result()
        return result_columns

    @reactive.calc
    def computation_matrices():
        _, _, _, _, score_matrices, _ = compute_task.result()
        return score_matrices

    @reactive.effect
    @reactive.event(input.decision_file, input.decision_separator)
    def auto_detect_decision_header():
        files = input.decision_file()
        if files:
            try:
                has_header = detect_csv_has_header(files[0]['datapath'], input.decision_separator(), '.')
                ui.update_checkbox('decision_header', value=has_header)
            except OSError:
                pass

    @reactive.calc
    def decision_data():
        try:
            files = input.decision_file()
            if not files:
                return pd.DataFrame(), None
            validate_csv_upload(files[0]['datapath'])
            frame = pd.read_csv(files[0]['datapath'], sep=input.decision_separator(),
                                header=0 if input.decision_header() else None,
                                dtype=str, keep_default_na=False)
            frame.columns = [str(column) for column in frame.columns]
            return frame, None
        except (ValueError, OSError, UnicodeError) as error:
            return pd.DataFrame(), str(error)

    @reactive.effect
    def decision_columns():
        frame, error = decision_data()
        columns = list(frame.columns)
        ui.update_select('decision_column', choices=columns, selected=columns[0] if columns else None)

    @reactive.effect
    def input_decision_columns():
        df, error = data()
        columns = list(df.columns)
        preferred = next((c for c in columns if c.lower() in ('label', 'decision', 'class', 'target', 'y')),
                         columns[-1] if columns else None)
        ui.update_select('input_decision_column', choices=columns, selected=preferred)

    @reactive.effect
    def exclude_decision_column():
        if input.decision_source() == 'input_column':
            col = input.input_decision_column()
            with reactive.isolate():
                current = selected_columns()
                if col and col in current:
                    selected_columns.set([c for c in current if c != col])

    @reactive.calc
    def optimal_classification():
        if input.decision_source() == 'none':
            return None, None
        try:
            result, source, error = computation()
            cols = computation_columns()
            matrices = computation_matrices()
            if error:
                raise ValueError('Correct the input data and aggregation parameters first. ' + error)
            if not cols:
                raise ValueError('No aggregation result columns found.')
            if input.decision_source() == 'input_column':
                if matrices is not None:
                    raise ValueError('In per-class CSV mode, paste labels or upload a separate label CSV.')
                df, data_error = data()
                if data_error:
                    raise ValueError(data_error)
                col = input.input_decision_column()
                if df.empty or col not in df.columns:
                    raise ValueError('Load an input CSV file and select the decision column.')
                decisions = df[col].astype(str).tolist()
            elif input.decision_source() == 'text':
                decisions = text_decisions(input.decision_text())
                if not decisions:
                    raise ValueError('Paste the label vector.')
            else:
                frame, error = decision_data()
                if error:
                    raise ValueError(error)
                if frame.empty or input.decision_column() not in frame.columns:
                    raise ValueError('Upload a CSV file and select the label column.')
                decisions = frame[input.decision_column()].tolist()
            if matrices is not None:
                class_names = per_class_data()[1]
                evaluations = {
                    col: evaluate_proba(decisions, class_names, matrices[col], input.positive_label())
                    for col in cols
                }
            else:
                evaluations = {
                    col: evaluate(decisions, result[col].to_numpy(), input.positive_label(),
                                  input.score_direction(), input.threshold_criterion())
                    for col in cols
                }
            return evaluations, None
        except (ValueError, TypeError) as error:
            return None, str(error)

    @reactive.calc
    def inspected_column():
        evaluations, _ = optimal_classification()
        if not evaluations:
            return None
        cols = list(evaluations.keys())
        try:
            selected = input.eval_method()
            if selected in evaluations:
                return selected
        except Exception:
            pass
        return cols[0]

    @render.ui
    def eval_method_ui():
        evaluations, _ = optimal_classification()
        if not evaluations or len(evaluations) <= 1:
            return None
        cols = list(evaluations.keys())
        return ui.input_select('eval_method', 'Inspected method (confusion matrices & manual threshold)',
                               cols, selected=cols[0], width='100%')

    @reactive.effect
    async def sync_eval_mode():
        evaluations, _ = optimal_classification()
        is_multiclass = False
        if evaluations:
            first = next(iter(evaluations.values()))
            is_multiclass = first.get('mode') == 'multiclass' or bool(first.get('fusion'))
        await session.send_custom_message('eval-mode', {'multiclass': is_multiclass})

    @reactive.effect
    def initialize_threshold():
        evaluations, error = optimal_classification()
        col = inspected_column()
        if evaluations and col:
            evaluation = evaluations[col]
            if evaluation.get('mode') == 'binary' and input.threshold_mode() == 'auto':
                value = evaluation['optimal_threshold']
                ui.update_numeric('manual_threshold', value=value if math.isfinite(value) else .5)

    @reactive.effect
    @reactive.event(input.reset_threshold)
    def reset_threshold():
        ui.update_select('threshold_criterion', selected='youden')
        ui.update_radio_buttons('threshold_mode', selected='auto')

    @reactive.calc
    def classification():
        evaluations, error = optimal_classification()
        if evaluations is None:
            return None, error
        col = inspected_column()
        active = evaluations[col]
        if active.get('mode') == 'multiclass' or active.get('fusion') or input.threshold_mode() == 'auto':
            return evaluations, None
        try:
            if input.manual_threshold() is None:
                raise ValueError('Enter a threshold t.')
            updated = dict(evaluations)
            updated[col] = evaluate(
                active['labels'], computation()[0][col].to_numpy(),
                input.positive_label(), input.score_direction(), input.threshold_criterion(),
                threshold=input.manual_threshold(),
            )
            return updated, None
        except (ValueError, TypeError) as err:
            return None, str(err)

    @render.ui
    def evaluation_status():
        evaluations, error = classification()
        if error:
            return ui.div(error, class_='alert alert-danger', role='status')
        if evaluations:
            col = inspected_column()
            evaluation = evaluations[col]
            prefix = f'[{col}] ' if len(evaluations) > 1 else ''
            if evaluation.get('mode') == 'multiclass':
                classes = ', '.join(evaluation['classes'])
                return ui.div(
                    f"{prefix}Multiclass evaluation ({len(evaluation['classes'])} classes: {classes}). "
                    f"Classification rule: {evaluation['rule']}.",
                    class_='alert alert-info',
                )
            if evaluation.get('fusion') and evaluation.get('mode') == 'binary':
                return ui.div(
                    f"{prefix}Per-class fusion ({evaluation['rule']}). "
                    f"Youden J = {evaluation['metrics']['Youden J']:.6g}, "
                    f"ROC AUC = {evaluation['metrics']['ROC AUC']:.6g}.",
                    class_='alert alert-info',
                )
            return ui.div(f"{prefix}Positive class rule ({evaluation['positive']}): {evaluation['rule']}. "
                          f"Youden J at the current threshold = {evaluation['metrics']['Youden J']:.6g}. "
                          f"Automatic threshold t = {evaluation['optimal_threshold']:.12g} "
                          f"({CRITERIA[input.threshold_criterion()]}). Ties favor the threshold that yields fewer positive predictions.",
                          class_='alert alert-info')

    @render.ui
    def roc_plot_ui():
        if classification()[0] is not None:
            return ui.output_plot('roc_plot', height='380px')

    @render.plot
    def roc_plot():
        evaluations, error = classification()
        if evaluations is None:
            return None
        col = inspected_column()
        evaluation = evaluations[col]
        figure = Figure(figsize=(7, 4), layout='tight')
        ax = figure.subplots()
        if evaluation.get('mode') == 'multiclass':
            from sklearn import metrics as sk_metrics
            classes = evaluation['classes']
            proba = evaluation['proba']
            y = evaluation['y']
            if evaluation.get('fusion'):
                for idx, cls in enumerate(classes):
                    fpr_c, tpr_c, _ = sk_metrics.roc_curve((y == idx).astype(int), proba[:, idx])
                    auc_c = sk_metrics.auc(fpr_c, tpr_c)
                    ax.plot(fpr_c, tpr_c, label=f"{cls} vs rest (AUC = {auc_c:.3f})")
                ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
                ax.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)',
                       xlim=(-.02, 1.02), ylim=(-.02, 1.02),
                       title=f"OvR ROC ({col})" if len(evaluations) > 1 else 'Multiclass OvR ROC curves')
                ax.legend(loc='lower right')
                return figure
            scores = evaluation['scores']
            figure = Figure(figsize=(10, 4), layout='tight')
            ax1, ax2 = figure.subplots(1, 2)
            grouped = [scores[y == idx] for idx in range(len(classes))]
            ax1.boxplot(grouped, tick_labels=[textwrap.fill(str(c), 14) for c in classes], vert=False)
            for idx, threshold in enumerate(evaluation['thresholds']):
                ax1.axvline(threshold, color='red', linestyle='--', linewidth=1.2,
                            label='Split threshold' if idx == 0 else None)
            title = f'Score distribution ({col})' if len(evaluations) > 1 else 'Score distribution by true class'
            ax1.set(xlabel='Aggregation score', ylabel='True class', title=title)
            if evaluation['thresholds']:
                ax1.legend(loc='best')

            for idx, cls in enumerate(classes):
                fpr_c, tpr_c, _ = sk_metrics.roc_curve((y == idx).astype(int), proba[:, idx])
                auc_c = sk_metrics.auc(fpr_c, tpr_c)
                ax2.plot(fpr_c, tpr_c, label=f"{cls} vs rest (AUC = {auc_c:.3f})")
            ax2.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax2.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)',
                    xlim=(-.02, 1.02), ylim=(-.02, 1.02),
                    title=f"OvR ROC ({col})" if len(evaluations) > 1 else 'Multiclass OvR ROC curves')
            ax2.legend(loc='lower right')
            return figure
        if len(evaluations) > 1:
            for name, ev in evaluations.items():
                line, = ax.plot(ev['fpr'], ev['tpr'], label=f"{name} (AUC = {ev['metrics']['ROC AUC']:.4f})")
                ax.scatter(*ev['point'], color=line.get_color(), zorder=3)
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)',
                   xlim=(-.02, 1.02), ylim=(-.02, 1.02), title='ROC curves comparison')
            ax.legend(loc='lower right')
            return figure
        ax.plot(evaluation['fpr'], evaluation['tpr'], label=f"ROC AUC = {evaluation['metrics']['ROC AUC']:.4f}")
        ax.scatter(*evaluation['point'], color='red', label=f"Threshold t = {evaluation['threshold']:.6g}", zorder=3)
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)', xlim=(-.02, 1.02), ylim=(-.02, 1.02), title='ROC curve for aggregation scores')
        ax.legend(loc='lower right')
        return figure

    @render.ui
    def metric_table():
        evaluations, error = classification()
        if evaluations is None:
            return None
        cols = list(evaluations.keys())
        first_metrics = evaluations[cols[0]]['metrics']
        rows = []
        for name in first_metrics:
            function, description = METRIC_DOCS[name]
            link = ui.tags.a(name,
                href=f'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.{function}.html',
                target='_blank', rel='noopener noreferrer', title=description)
            vals = [evaluations[c]['metrics'][name] for c in cols]
            best = max(vals) if len(cols) > 1 else None
            cells = [
                ui.tags.td(
                    ui.tags.strong(f'{v:.6g}') if (best is not None and math.isclose(v, best)) else f'{v:.6g}'
                )
                for v in vals
            ]
            rows.append(ui.tags.tr(ui.tags.td(link), *cells))
        header_cells = [ui.tags.th(c, scope='col') for c in cols] if len(cols) > 1 else [ui.tags.th('Value', scope='col')]
        return ui.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(ui.tags.th('Measure', scope='col'), *header_cells)),
                ui.tags.tbody(*rows), class_='table table-striped table-hover mb-0'),
            class_='table-responsive',
        )

    @render.ui
    def confusion_plots_ui():
        if classification()[0] is not None:
            return ui.layout_columns(
                ui.output_plot('confusion_counts', height='420px'),
                ui.output_plot('confusion_rates', height='420px'),
            )

    @render.plot
    def confusion_counts():
        evaluations, error = classification()
        if not evaluations:
            return None
        return confusion_heatmap(evaluations[inspected_column()])

    @render.plot
    def confusion_rates():
        evaluations, error = classification()
        if not evaluations:
            return None
        return confusion_heatmap(evaluations[inspected_column()], normalized=True)

    @render.ui
    def classification_download_ui():
        if classification()[0] is not None:
            return ui.download_button('classification_csv', 'Download labels, scores, and predictions (CSV)',
                                      class_='btn-sm btn-outline-primary')

    @render.download_button(filename='classification.csv', media_type='text/csv')
    def classification_csv():
        evaluations, error = classification()
        if evaluations:
            col = inspected_column()
            evaluation = evaluations[col]
            score_col = evaluation['proba'].max(axis=1) if evaluation.get('fusion') else computation()[0][col].to_numpy()
            frame = pd.DataFrame({'decision': evaluation['labels'],
                'score': score_col,
                'prediction': evaluation['prediction_labels'], 'threshold': evaluation['threshold']})
            yield frame.to_csv(index=False, sep=input.separator(), decimal=input.decimal()).encode('utf-8-sig')

    @render.data_frame
    def preview():
        return preview_grid(data()[0], input.preview_view() == 'all', sort_states().get('preview'))

    @render.ui
    def status():
        task_status = compute_task.status()
        if task_status == 'running':
            reactive.invalidate_later(0.15)
            phase = progress_state['phase']
            done = progress_state['done']
            total = progress_state['total']
            if phase == 'queued':
                label = f'Waiting in queue… (active: {_ACTIVE_TASKS}/{MAX_WORKERS}, queued: {_QUEUED_TASKS})'
                bar = ui.div(
                    ui.div(class_='progress-bar progress-bar-striped progress-bar-animated w-100',
                           role='progressbar'),
                    class_='progress mt-2', style='height: 10px;',
                )
                return ui.div(ui.div(label), bar, class_='alert alert-info mb-0', role='status')
            pct = int(round(100 * done / total)) if total > 0 else 0
            label = f'Calculating… {done:,} / {total:,} rows ({pct}%)'
            bar = ui.div(
                ui.div(class_='progress-bar progress-bar-striped progress-bar-animated',
                       role='progressbar',
                       style=f'width: {pct}%;',
                       aria_valuenow=str(pct), aria_valuemin='0', aria_valuemax='100'),
                class_='progress mt-2', style='height: 10px;',
            )
            return ui.div(ui.div(label), bar, class_='alert alert-info mb-0', role='status')

        if task_status == 'initial':
            return ui.div('Select columns and click Run aggregation.', class_='alert alert-secondary mb-0', role='status')

        result, code, error = computation()
        sig, _, _ = current_request()
        stale = last_completed_sig() is not None and sig != last_completed_sig()
        alert = ui.div(error if error else f'Processed {len(result):,} rows.',
                       class_='alert alert-danger mb-0' if error else 'alert alert-success mb-0', role='status')
        if stale:
            return ui.div(
                alert,
                ui.div('Parameters or data changed — click Run aggregation to update results.',
                       class_='alert alert-warning mt-2 mb-0', role='status'),
            )
        return alert

    @render.data_frame
    def results():
        result, code, error = computation()
        return preview_grid(result if result is not None else pd.DataFrame(),
                            input.results_view() == 'all', sort_states().get('results'))

    @render.ui
    def code():
        source = computation()[1]
        if source:
            evaluations, error = classification()
            if evaluations:
                result_name = inspected_column()
                evaluation = evaluations[result_name]
                if evaluation.get('fusion'):
                    return ui.HTML(highlight(source, PythonLexer(), CODE_FORMATTER))
                if evaluation.get('mode') == 'multiclass':
                    source += (
                        '\n# Multiclass evaluation; labels follow original row order.\n'
                        'from sklearn import metrics\n'
                        'from sklearn.tree import DecisionTreeClassifier\n'
                        'import matplotlib.pyplot as plt\n'
                        f"classes = {evaluation['classes']!r}\n"
                        f"y_true = np.array({evaluation['labels'].tolist()!r})\n"
                        f"score = df[{result_name!r}].to_numpy()\n"
                        'tree = DecisionTreeClassifier(max_leaf_nodes=len(classes), random_state=0)\n'
                        'tree.fit(score.reshape(-1, 1), y_true)\n'
                        'y_pred = tree.predict(score.reshape(-1, 1))\n'
                        "print('ACC:', metrics.accuracy_score(y_true, y_pred))\n"
                        "print('Balanced ACC:', metrics.balanced_accuracy_score(y_true, y_pred))\n"
                        "for average in ('micro', 'macro', 'weighted'):\n"
                        "    print('Precision', average, metrics.precision_score(y_true, y_pred, average=average, zero_division=0))\n"
                        "    print('Recall', average, metrics.recall_score(y_true, y_pred, average=average, zero_division=0))\n"
                        "    print('F1', average, metrics.f1_score(y_true, y_pred, average=average, zero_division=0))\n"
                        "print('MCC:', metrics.matthews_corrcoef(y_true, y_pred))\n"
                        "print('Cohen kappa:', metrics.cohen_kappa_score(y_true, y_pred))\n"
                        "print(metrics.classification_report(y_true, y_pred, zero_division=0))\n"
                        "fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='tight')\n"
                        "for ax, normalize in zip(axes, (None, 'true')):\n"
                        "    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=classes, normalize=normalize, cmap='Blues', ax=ax)\n"
                        "    ax.set_title('Counts' if normalize is None else 'Shares by true class')\n"
                        "plt.show()\n"
                    )
                    return ui.HTML(highlight(source, PythonLexer(), CODE_FORMATTER))
                objective = {
                    'youden': 'tp * (len(y_true) - y_true.sum()) - fp * y_true.sum()',
                    'accuracy': 'tp + tn',
                    'f1': '2 * tp / np.maximum(2 * tp + fp + fn, 1)',
                }[input.threshold_criterion()]
                cutoff_code = 'thresholds[best]'
                if input.threshold_mode() == 'manual':
                    cutoff_code = repr(evaluation['threshold'] if input.score_direction() == 'higher' else -evaluation['threshold'])
                source += (
                    '\n# Binary evaluation; labels follow original row order.\n'
                    'from sklearn import metrics\nimport matplotlib.pyplot as plt\n'
                    f"y_true = np.array({evaluation['y'].tolist()!r})\n"
                    f"score = df[{result_name!r}].to_numpy()\n"
                    + ('oriented = score\n' if input.score_direction() == 'higher' else 'oriented = -score\n')
                    + 'fpr, tpr, thresholds = metrics.roc_curve(y_true, oriented, drop_intermediate=False)\n'
                    'tp = np.rint(tpr * y_true.sum())\nfp = np.rint(fpr * (len(y_true) - y_true.sum()))\n'
                    'fn = y_true.sum() - tp\ntn = len(y_true) - y_true.sum() - fp\n'
                    f'best = int(np.argmax({objective}))\n'
                    f'cutoff = {cutoff_code}\n'
                    'y_pred = (oriented >= cutoff).astype(int)\n'
                    + ('threshold = cutoff\n' if input.score_direction() == 'higher' else 'threshold = -cutoff\n')
                    + "print('Threshold:', threshold)\nprint('ROC AUC:', metrics.roc_auc_score(y_true, oriented))\n"
                    "print('ACC:', metrics.accuracy_score(y_true, y_pred))\n"
                    "print('Balanced ACC:', metrics.balanced_accuracy_score(y_true, y_pred))\n"
                    "for average in ('micro', 'macro', 'weighted'):\n    print('F1', average, metrics.f1_score(y_true, y_pred, average=average, zero_division=0))\n"
                    "print(metrics.classification_report(y_true, y_pred, zero_division=0))\n"
                    "print(metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]))\n"
                    "tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()\n"
                    "print('Youden J:', tp / (tp + fn) - fp / (fp + tn))\n"
                    "plt.plot(fpr, tpr, label='ROC')\nplt.scatter(fp / (fp + tn), tp / (tp + fn), label='Chosen threshold')\n"
                    "plt.xlabel('FPR')\nplt.ylabel('TPR')\nplt.legend()\nplt.show()\n"
                    "fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='tight')\n"
                    "for ax, normalize in zip(axes, (None, 'true')):\n"
                    "    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], normalize=normalize, cmap='Blues', ax=ax)\n"
                    "    ax.set_title('Counts' if normalize is None else 'Shares by true class')\n"
                    "plt.show()\n"
                )
            # Pygments escapes source text before generating HTML markup.
            return ui.HTML(highlight(source, PythonLexer(), CODE_FORMATTER))

    @render.ui
    def download_ui():
        if computation()[0] is not None:
            return ui.download_button('download', 'Download results (CSV)', class_='btn-sm btn-outline-primary')

    @render.download_button(filename='results.csv', media_type='text/csv')
    def download():
        result, code, error = computation()
        if result is not None:
            yield result.to_csv(index=False, sep=input.separator(), decimal=input.decimal()).encode('utf-8-sig')


def confusion_heatmap(evaluation, normalized=False):
    matrix = evaluation['normalized' if normalized else 'matrix']
    n = matrix.shape[0]
    figure = Figure(figsize=(max(5, 1.2 * n), max(4.4, 1.05 * n)), layout='tight')
    ax = figure.subplots()
    maximum = 1. if normalized else max(1, int(matrix.max()))
    image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum)
    if evaluation.get('mode') == 'multiclass':
        labels = [textwrap.fill(str(c), 18) for c in evaluation['classes']]
        title_suffix = ''
        cell_names = None
    else:
        labels = [textwrap.fill(f"− ({evaluation['negative']})", 18),
                  textwrap.fill(f"+ ({evaluation['positive']})", 18)]
        title_suffix = f"\nt = {evaluation['threshold']:.6g}"
        cell_names = (('TN', 'FP'), ('FN', 'TP'))
    ticks = list(range(n))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels,
           xlabel='Predicted class', ylabel='True class',
           title=('Confusion matrix (percentages by true class)' if normalized else 'Confusion matrix (counts)') + title_suffix)
    if n > 3:
        ax.tick_params(axis='x', rotation=25)
    for row in range(n):
        for column in range(n):
            value = matrix[row, column]
            formatted = f'{value:.1%}' if normalized else str(int(value))
            text = f'{cell_names[row][column]}\n{formatted}' if cell_names else formatted
            ax.text(column, row, text, ha='center', va='center',
                    color='white' if value > maximum / 2 else '#172b4d',
                    fontsize=12 if n <= 3 else 10)
    colorbar = figure.colorbar(image, ax=ax, fraction=.046, pad=.04)
    if normalized:
        from matplotlib.ticker import PercentFormatter
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        from matplotlib.ticker import MaxNLocator
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return figure


def preview_grid(frame, full=False, sort=None):
    preview = grid_view(frame, full=full, sort=sort)
    return render.DataGrid(preview, width='100%', height=None if len(preview) <= 8 else '390px',
                           summary=False, filters=False, selection_mode='none')


app = App(app_ui, server)
