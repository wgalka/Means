"""Run from the repository root: python -m shiny run apps/shiny/app.py"""
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

from apps.shiny.core import AGGREGATIONS, MEASURES, PARAMETERS, calculate, header_weight_vector, grid_view
from apps.shiny.evaluation import CRITERIA, METRIC_DOCS, evaluate, text_decisions

EXAMPLE = Path(__file__).with_name('example.csv')
CODE_FORMATTER = HtmlFormatter(cssclass='python-code', style='friendly')


def view_picker(id, label):
    return ui.div(ui.input_radio_buttons(id, label,
                  {'short': '5 … 5', 'all': 'All rows'}, selected='short', inline=True),
                  class_='view-picker', title='Compact view shows the first and last five rows. Full view shows all rows.')


app_ui = ui.page_fluid(
    ui.div(
    ui.h2('aggregationslib'),
    ui.p('Upload your data, choose an aggregation method, and download the results. Add ground truth labels to evaluate classification.', class_='text-muted'),
    ui.tags.style('.container-fluid {max-width: 1600px;} pre {max-height: 500px; overflow: auto;} '
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
                  '\n.python-code {padding: 1rem; border-radius: 6px;} '
                  '.python-code pre {margin: 0; white-space: pre;}'),
    ui.tags.script(ui.HTML(Path(__file__).with_name('column-selection.js').read_text(encoding='utf-8'))),

    ui.tags.style('''
        body {background: #f5f7fa;}
        .app-shell {max-width: 1120px; margin: 2rem auto; padding: 0 1rem;
                    display: flex; flex-direction: column; gap: 1.25rem;}
        .app-shell > h2, .app-shell > p {margin-bottom: 0;}
        .app-shell .card {box-shadow: 0 2px 8px #18263c08;}
        .app-shell .card-header {font-weight: 600; padding: 1rem 1.25rem;}
        .app-shell .shiny-input-container {max-width: 100%;}
    '''),
    ui.card(ui.card_header('1. Input data'),
        ui.layout_columns(
            ui.div(ui.input_file('file', 'Upload a CSV file', accept=['.csv'], width='100%'),
                   ui.help_text('Each row contains one vector to aggregate.')),
            ui.div(ui.input_select('separator', 'CSV delimiter', {',': 'Comma', ';': 'Semicolon', '\t': 'Tab'}),
                   ui.input_select('decimal', 'Decimal separator', {'.': 'Period', ',': 'Comma'}),
                   ui.input_checkbox('header', 'First row contains column headers', True))),
        ui.accordion(ui.accordion_panel('Additional vectors (optional)',
            ui.input_text_area('extra', 'Vectors to append', width='100%', placeholder='0.1 0.2 0.3\n0.4 0.5 0.6'),
            ui.help_text('Enter one vector per line. Separate numbers with spaces and use a decimal point. Enter values in the order of the selected columns.')),
            open=False)),
    ui.card(ui.card_header('2. Ground truth labels (optional)'),
            ui.input_radio_buttons('decision_source', 'Ground truth labels',
                {'none': 'No labels', 'text': 'Paste labels', 'csv': 'Upload CSV'}, inline=True),
            ui.panel_conditional("input.decision_source === 'text'",
                ui.input_text_area('decision_text', 'Label vector', width='100%', placeholder='0 1 0 1 …')),
            ui.panel_conditional("input.decision_source === 'csv'",
                ui.input_file('decision_file', 'CSV with labels', accept=['.csv']),
                ui.input_select('decision_separator', 'Label CSV delimiter', {',': 'Comma', ';': 'Semicolon', '\t': 'Tab'}),
                ui.input_checkbox('decision_header', 'Label CSV has a header row', True),
                ui.input_select('decision_column', 'Label column', [])),

        ui.panel_conditional("input.decision_source !== 'none'",
            ui.help_text('Enter one label per row in the original order, followed by labels for any additional vectors.'),
            ui.layout_columns(
                ui.input_text('positive_label', 'Positive class label', '1'),
                ui.input_select('score_direction', 'Predict positive when the score is', {'higher': 'At or above the threshold', 'lower': 'At or below the threshold'})))),
    ui.card(ui.card_header('3. Aggregation and parameters'),
        ui.input_select('aggregation', 'Aggregation method', AGGREGATIONS, selected='A_ar', width='100%'),
        ui.panel_conditional("['A_pw','A_ex','A_lm'].includes(input.aggregation)",
                             ui.input_numeric('r', 'Parameter r', 2)),
        ui.panel_conditional("['A_ex2','A_ex3','A_amn','A_amx','A_oln'].includes(input.aggregation)",
                             ui.input_numeric('p', 'Parameter p', 1)),
        ui.panel_conditional("['A_ex2','A_ex3'].includes(input.aggregation)",
                             ui.input_numeric('q', 'Parameter q', 2)),
        ui.panel_conditional("['A_ch','A_sg'].includes(input.aggregation)",
            ui.input_select('measure', 'Measure', MEASURES),
            ui.panel_conditional("input.measure === 'exponential_cardinality'",
                ui.input_numeric('measure_q', 'Measure parameter q > 0', 1, min=0, step=.1)),
            ui.help_text('The integrals accept values in [0, 1]. With the uniform measure, the Choquet integral equals the arithmetic mean. The first OWA weight applies to the largest value.')),
    ),
    ui.card(ui.card_header('4. Columns and weights'),
            ui.div(ui.input_action_button('select_all', 'Select all'),
                   ui.input_action_button('select_none', 'Deselect all'),
                   class_='d-flex gap-2'),
            ui.help_text('The default view shows the first and last five rows. Use the header checkboxes to select columns for aggregation. Click a column name to sort the full table. Enter weights and additional vectors in file column order. The CSV export keeps the original row order.'),
            ui.output_ui('weight_hint'),
            view_picker('preview_view', 'Input view'), ui.output_data_frame('preview')),

    ui.output_ui('status'),
    ui.card(ui.card_header('5. Aggregation results'),
            ui.help_text('The default view shows the first and last five rows. Sorting applies to all results.'),
            view_picker('results_view', 'Results view'),
            ui.output_data_frame('results'), ui.output_ui('download_ui')),

    ui.panel_conditional("input.decision_source !== 'none'",
        ui.card(ui.card_header('6. Classification evaluation and threshold'),
            ui.input_select('threshold_criterion', 'Threshold criterion', CRITERIA),
                ui.input_radio_buttons('threshold_mode', 'Threshold t',
                    {'auto': 'Automatic', 'manual': 'Manual'}, selected='auto', inline=True),
                ui.panel_conditional("input.threshold_mode === 'manual'",
                    ui.input_numeric('manual_threshold', 'Custom threshold t', .5, step=.01)),
                ui.input_action_button('reset_threshold', 'Reset to Youden threshold'),
                ui.output_ui('evaluation_status'),
                ui.output_plot('roc_plot', height='380px'),
                ui.output_ui('metric_table'),
                ui.layout_columns(
                    ui.output_plot('confusion_counts', height='420px'),
                    ui.output_plot('confusion_rates', height='420px')),
                ui.help_text('This report selects and evaluates the threshold on the same data. To assess generalization, choose a threshold on validation data and evaluate it on a separate test set. Metrics with a zero denominator are set to 0.'),
                ui.output_ui('classification_download_ui'))),
    ui.card(ui.card_header('Run this in Python'), ui.p('If you uploaded a file, save a local copy as input.csv.'),
            ui.output_ui('code')),

    class_='app-shell'),
    title='aggregationslib · Table aggregation',
)



def server(input, output, session):
    selected_columns = reactive.value([])
    weight_values = reactive.value({})
    sort_states = reactive.value({})

    @reactive.calc
    def tester_mode():
        query = parse_qs(session.clientdata.url_search().lstrip('?'))
        return query.get('tester') == ['1']

    @reactive.effect
    def load_test_decisions():
        if tester_mode():
            frame = pd.read_csv(EXAMPLE)
            decisions = (frame.mean(axis=1) >= .5).astype(int)
            ui.update_text_area('decision_text', value=' '.join(map(str, decisions)))
            ui.update_radio_buttons('decision_source', selected='text')

    @reactive.calc
    def data():
        try:
            file = input.file()
            if file:
                if input.separator() == input.decimal():
                    raise ValueError('The CSV delimiter and decimal separator must be different.')
                df = pd.read_csv(file[0]['datapath'], sep=input.separator(),
                                 decimal=input.decimal(), header=0 if input.header() else None)
                if not input.header():
                    df.columns = [f'x{i + 1}' for i in range(len(df.columns))]
                return df, None
            if tester_mode():
                return pd.read_csv(EXAMPLE), None
            return pd.DataFrame(), 'Upload a CSV file to get started.'
        except (ValueError, OSError, UnicodeError) as error:
            return pd.DataFrame(), str(error)

    @reactive.effect
    def update_columns():
        df, error = data()
        selected_columns.set(list(df.columns))
        weight_values.set({})
        sort_states.set({})

    @reactive.calc
    def weight_kind():
        if input.aggregation() in ('A_ch', 'A_sg') and input.measure() in ('weighted', 'owa', 'k_fuzzy'):
            return input.measure()
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

    @reactive.calc
    def computation():
        df, error = data()
        if error:
            return None, '', error
        try:
            name = input.aggregation()
            params = {key: getattr(input, key)() for key in PARAMETERS.get(name, ())}
            if name in ('A_ch', 'A_sg') and input.measure() == 'exponential_cardinality':
                params['measure_q'] = input.measure_q()
            columns = [column for column in df.columns if column in selected_columns()]
            kind = weight_kind()
            weights = header_weight_vector(kind, columns, weight_values().get(kind, {})) if kind else ''
            result, code = calculate(df, columns, input.extra(), name, params,
                                     input.measure(), weights, input.separator(),
                                     input.decimal(), input.header(), bool(input.file()))
            return result, code, None
        except (ValueError, TypeError, ArithmeticError, KeyError) as error:
            return None, '', str(error)

    @reactive.calc
    def decision_data():
        try:
            files = input.decision_file()
            if not files:
                return pd.DataFrame(), None
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

    @reactive.calc
    def optimal_classification():
        if input.decision_source() == 'none':
            return None, None
        try:
            result, source, error = computation()
            if error:
                raise ValueError('Correct the input data and aggregation parameters first. ' + error)
            if input.decision_source() == 'text':
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
            return evaluate(decisions, result.iloc[:, -1].to_numpy(), input.positive_label(),
                            input.score_direction(), input.threshold_criterion()), None
        except (ValueError, TypeError) as error:
            return None, str(error)

    @reactive.effect
    def initialize_threshold():
        evaluation, error = optimal_classification()
        if evaluation and input.threshold_mode() == 'auto':
            value = evaluation['optimal_threshold']
            ui.update_numeric('manual_threshold', value=value if math.isfinite(value) else .5)

    @reactive.effect
    @reactive.event(input.reset_threshold)
    def reset_threshold():
        ui.update_select('threshold_criterion', selected='youden')
        ui.update_radio_buttons('threshold_mode', selected='auto')

    @reactive.calc
    def classification():
        evaluation, error = optimal_classification()
        if evaluation is None or input.threshold_mode() == 'auto':
            return evaluation, error
        try:
            if input.manual_threshold() is None:
                raise ValueError('Enter a threshold t.')
            return evaluate(evaluation['labels'], computation()[0].iloc[:, -1].to_numpy(),
                            input.positive_label(), input.score_direction(), input.threshold_criterion(),
                            threshold=input.manual_threshold()), None
        except (ValueError, TypeError) as error:
            return None, str(error)

    @render.ui
    def evaluation_status():
        evaluation, error = classification()
        if error:
            return ui.div(error, class_='alert alert-danger', role='status')
        if evaluation:
            return ui.div(f"Positive class rule ({evaluation['positive']}): {evaluation['rule']}. "
                          f"Youden J at the current threshold = {evaluation['metrics']['Youden J']:.6g}. "
                          f"Automatic threshold t = {evaluation['optimal_threshold']:.12g} "
                          f"({CRITERIA[input.threshold_criterion()]}). Ties favor the threshold that yields fewer positive predictions.",
                          class_='alert alert-info')

    @render.plot
    def roc_plot():
        evaluation, error = classification()
        if evaluation is None:
            return None
        figure = Figure(figsize=(7, 4), layout='tight')
        ax = figure.subplots()
        ax.plot(evaluation['fpr'], evaluation['tpr'], label=f"ROC AUC = {evaluation['metrics']['ROC AUC']:.4f}")
        ax.scatter(*evaluation['point'], color='red', label=f"Threshold t = {evaluation['threshold']:.6g}", zorder=3)
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        ax.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)', xlim=(-.02, 1.02), ylim=(-.02, 1.02), title='ROC curve for aggregation scores')
        ax.legend(loc='lower right')
        return figure

    @render.ui
    def metric_table():
        evaluation, error = classification()
        if evaluation is None:
            return None
        rows = []
        for name, value in evaluation['metrics'].items():
            function, description = METRIC_DOCS[name]
            link = ui.tags.a(name,
                href=f'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.{function}.html',
                target='_blank', rel='noopener noreferrer', title=description)
            rows.append(ui.tags.tr(ui.tags.td(link), ui.tags.td(f'{value:.6g}')))
        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(ui.tags.th('Measure', scope='col'), ui.tags.th('Value', scope='col'))),
            ui.tags.tbody(*rows), class_='table table-striped table-hover')

    @render.plot
    def confusion_counts():
        evaluation, error = classification()
        return confusion_heatmap(evaluation) if evaluation else None

    @render.plot
    def confusion_rates():
        evaluation, error = classification()
        return confusion_heatmap(evaluation, normalized=True) if evaluation else None

    @render.ui
    def classification_download_ui():
        if classification()[0] is not None:
            return ui.download_button('classification_csv', 'Download labels, scores, and predictions (CSV)')

    @render.download_button(filename='classification.csv', media_type='text/csv')
    def classification_csv():
        evaluation, error = classification()
        if evaluation:
            frame = pd.DataFrame({'decision': evaluation['labels'],
                'score': computation()[0].iloc[:, -1].to_numpy(),
                'prediction': evaluation['prediction_labels'], 'threshold': evaluation['threshold']})
            yield frame.to_csv(index=False, sep=input.separator(), decimal=input.decimal()).encode('utf-8-sig')

    @render.data_frame
    def preview():
        return preview_grid(data()[0], input.preview_view() == 'all', sort_states().get('preview'))

    @render.ui
    def status():
        result, code, error = computation()
        return ui.div(error if error else f'Processed {len(result)} rows.',
                      class_='alert alert-danger' if error else 'alert alert-success', role='status')

    @render.data_frame
    def results():
        result, code, error = computation()
        return preview_grid(result if result is not None else pd.DataFrame(),
                            input.results_view() == 'all', sort_states().get('results'))

    @render.ui
    def code():
        source = computation()[1]
        if source:
            evaluation, error = classification()
            if evaluation:
                result_name = computation()[0].columns[-1]
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
            return ui.download_button('download', 'Download results (CSV)')

    @render.download_button(filename='results.csv', media_type='text/csv')
    def download():
        result, code, error = computation()
        if result is not None:
            yield result.to_csv(index=False, sep=input.separator(), decimal=input.decimal()).encode('utf-8-sig')


def confusion_heatmap(evaluation, normalized=False):
    matrix = evaluation['normalized' if normalized else 'matrix']
    figure = Figure(figsize=(5, 4.4), layout='tight')
    ax = figure.subplots()
    maximum = 1. if normalized else max(1, int(matrix.max()))
    image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum)
    labels = [textwrap.fill(f"− ({evaluation['negative']})", 18),
              textwrap.fill(f"+ ({evaluation['positive']})", 18)]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel='Predicted class', ylabel='True class',
           title=('Confusion matrix (percentages by true class)' if normalized else 'Confusion matrix (counts)')
                 + f"\nt = {evaluation['threshold']:.6g}")
    names = (('TN', 'FP'), ('FN', 'TP'))
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = f'{value:.1%}' if normalized else str(int(value))
            ax.text(column, row, f'{names[row][column]}\n{label}', ha='center', va='center',
                    color='white' if value > maximum / 2 else '#172b4d', fontsize=12)
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
    return render.DataGrid(preview, width='100%', height='390px',
                           summary=False, filters=False, selection_mode='none')


app = App(app_ui, server)
