"""Data preparation shared by the Shiny app and its generated examples."""
import csv
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

from aggregationslib import aggregation, ndarray


AGGREGATIONS = {
    'A_mn': 'Minimum', 'A_mx': 'Maximum', 'A_pr': 'Product',
    'A_ar': 'Arithmetic mean', 'A_qd': 'Quadratic mean',
    'A_gm': 'Geometric mean', 'A_hm': 'Harmonic mean',
    'A_pw': 'Power mean', 'A_ex': 'Exponential mean',
    'A_ex2': 'Exponential mean II', 'A_ex3': 'Exponential mean III',
    'A_lm': 'Lehmer mean (0/0 = 0 convention)',
    'A_lm1': 'Lehmer mean (continuous extension)', 'A_amn': 'Arithmetic minimum mean',
    'A_amx': 'Arithmetic maximum mean', 'A_md': 'Median',
    'A_ol': 'Olympic mean', 'A_oln': 'Generalized Olympic mean',
    'A_lo': 'Logarithmic mean', 'A_ch': 'Choquet integral',
    'A_sg': 'Sugeno integral',
}
PARAMETERS = {
    'A_pw': ('r',), 'A_ex': ('r',), 'A_lm': ('r',),
    'A_lm1': ('r',),
    'A_ex2': ('p', 'q'), 'A_ex3': ('p', 'q'),
    'A_amn': ('p',), 'A_amx': ('p',), 'A_oln': ('p',),
}
MEASURES = {'uniform': 'Uniform', 'relative': 'Relative',
            'exponential_cardinality': 'Exponential cardinality',
            'weighted': 'Weighted', 'owa': 'OWA', 'k_fuzzy': 'k-fuzzy (λ)'}
AGGREGATION_FORMULAS = {
    'A_mn': r'\min(x_1, \dots, x_n)',
    'A_mx': r'\max(x_1, \dots, x_n)',
    'A_pr': r'\prod_{k=1}^{n} x_k',
    'A_ar': r'\frac{1}{n}\sum_{k=1}^{n} x_k',
    'A_qd': r'\sqrt{\frac{1}{n}\sum_{k=1}^{n} x_k^2}',
    'A_gm': r'\left(\prod_{k=1}^{n} x_k\right)^{1/n}',
    'A_hm': r'\frac{n}{\sum_{k=1}^{n} \frac{1}{x_k}}',
    'A_pw': r'\left(\frac{1}{n}\sum_{k=1}^{n} x_k^r\right)^{1/r}',
    'A_ex': r'\frac{1}{r}\ln\left(\frac{1}{n}\sum_{k=1}^{n} e^{r x_k}\right)',
    'A_ex2': r'\log_q\left(\log_p\left(\frac{1}{n}\sum_{k=1}^{n} p^{q^{x_k}}\right)\right)',
    'A_ex3': r'\frac{1}{q}\log_p\left(\frac{1}{n}\sum_{k=1}^{n} p^{q x_k}\right)',
    'A_lm': r'\frac{\sum_{k=1}^{n} x_k^r}{\sum_{k=1}^{n} x_k^{r-1}}',
    'A_lm1': r'\frac{\sum_{k=1}^{n} x_k^r}{\sum_{k=1}^{n} x_k^{r-1}}',
    'A_amn': r'\frac{p}{n}\sum_{k=1}^{n} x_k + (1-p)\min_k x_k',
    'A_amx': r'\frac{p}{n}\sum_{k=1}^{n} x_k + (1-p)\max_k x_k',
    'A_md': r'\operatorname{median}(x_1, \dots, x_n)',
    'A_ol': r'\frac{1}{n-2}\sum_{k=2}^{n-1} y_k',
    'A_oln': r'\frac{1}{n-2p}\sum_{k=p+1}^{n-p} y_k',
    'A_lo': r'(n-1)!\sum_{i=1}^{n}\frac{x_i}{\prod_{j\neq i}\ln(x_i/x_j)}',
    'A_ch': r'\sum_{i=1}^{n}(x_{\sigma(i)}-x_{\sigma(i-1)})\,\mu(A_{\sigma(i)})',
    'A_sg': r'\max_{1\le i\le n}\min\big(x_{\sigma(i)},\,\mu(A_{\sigma(i)})\big)',
}
MEASURE_FORMULAS = {
    'uniform': r'\frac{n-i+1}{n}',
    'relative': r'\frac{\sum_{j=i}^{n}\sigma(j)}{\sum_{j=1}^{n} j}',
    'exponential_cardinality': r'\left(\frac{n-i+1}{n}\right)^{%s}',
    'weighted': r'\sum_{j=i}^{n} w_{\sigma(j)}',
    'owa': r'\sum_{j=1}^{n-i+1} w_j',
    'k_fuzzy': r'\frac{\prod_{j=i}^{n}(1+k w_{\sigma(j)})-1}{k}',
}


def _fmt_latex_num(val, fallback):
    if val is None:
        return fallback
    try:
        num = float(val)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(num):
        return fallback
    return str(int(num)) if num.is_integer() else f'{num:g}'


def format_method_latex(name, measure='uniform', params=None, weights=''):
    """Return an inline LaTeX expression with chosen parameters, measure, and weights substituted."""
    params = params or {}
    sub = name[2:] if name.startswith('A_') else name
    r = _fmt_latex_num(params.get('r'), 'r')
    p = _fmt_latex_num(params.get('p'), 'p')
    q = _fmt_latex_num(params.get('q'), 'q')

    if name == 'A_pw':
        lhs, rhs = rf'A_{{pw}}^{{({r})}}(x)', rf'\left(\frac{{1}}{{n}}\sum_{{k=1}}^{{n}} x_k^{{{r}}}\right)^{{1/{r}}}'
    elif name == 'A_ex':
        lhs, rhs = rf'A_{{ex}}^{{({r})}}(x)', rf'\frac{{1}}{{{r}}}\ln\left(\frac{{1}}{{n}}\sum_{{k=1}}^{{n}} e^{{{r} x_k}}\right)'
    elif name == 'A_ex2':
        lhs, rhs = rf'A_{{ex2}}^{{({p},{q})}}(x)', rf'\log_{{{q}}}\left(\log_{{{p}}}\left(\frac{{1}}{{n}}\sum_{{k=1}}^{{n}} {p}^{{{q}^{{x_k}}}}\right)\right)'
    elif name == 'A_ex3':
        lhs, rhs = rf'A_{{ex3}}^{{({p},{q})}}(x)', rf'\frac{{1}}{{{q}}}\log_{{{p}}}\left(\frac{{1}}{{n}}\sum_{{k=1}}^{{n}} {p}^{{{q} x_k}}\right)'
    elif name in ('A_lm', 'A_lm1'):
        r_minus_1 = _fmt_latex_num(float(params['r']) - 1, f'{r}-1') if params.get('r') is not None else f'{r}-1'
        lhs = rf'A_{{{sub}}}^{{({r})}}(x)'
        quotient = rf'\frac{{\sum_{{k=1}}^{{n}} x_k^{{{r}}}}}{{\sum_{{k=1}}^{{n}} x_k^{{{r_minus_1}}}}}'
        zero_condition = 'r<1' if name == 'A_lm1' else r'r\le 1'
        arithmetic_condition = 'r=1' if name == 'A_lm1' else r'r=1,\ \forall k:\ x_k>0'
        rhs = (r'\begin{cases}'
               rf'0 & {zero_condition},\ \exists k:\ x_k=0\\'
               rf'\frac{{1}}{{n}}\sum_{{k=1}}^n x_k & {arithmetic_condition}\\'
               r'0 & r>1,\ x_1=\dots=x_n=0\\'
               rf'{quotient} & \text{{otherwise}}'
               r'\end{cases}')
    elif name in ('A_amn', 'A_amx'):
        one_minus_p = _fmt_latex_num(1 - float(params['p']), f'(1-{p})') if params.get('p') is not None else f'(1-{p})'
        op = r'\min_k' if name == 'A_amn' else r'\max_k'
        lhs, rhs = rf'A_{{{sub}}}^{{({p})}}(x)', rf'\frac{{{p}}}{{n}}\sum_{{k=1}}^{{n}} x_k + {one_minus_p}\,{op} x_k'
    elif name == 'A_oln':
        if params.get('p') is not None:
            p_val = int(round(float(params['p'])))
            two_p = str(2 * p_val)
            p_plus_1 = str(p_val + 1)
            p_str = str(p_val)
        else:
            two_p, p_plus_1, p_str = '2p', 'p+1', 'p'
        lhs, rhs = rf'A_{{oln}}^{{({p_str})}}(x)', rf'\frac{{1}}{{n-{two_p}}}\sum_{{k={p_plus_1}}}^{{n-{p_str}}} y_k'
    elif name in ('A_ch', 'A_sg'):
        m_tex = MEASURE_FORMULAS.get(measure, r'\mu(A_{\sigma(i)})')
        if measure == 'exponential_cardinality':
            mq = _fmt_latex_num(params.get('measure_q'), 'q')
            m_tex = m_tex % mq
        lhs = rf'A_{{{sub}}}(x)'
        if name == 'A_ch':
            rhs = rf'\sum_{{i=1}}^{{n}}(x_{{\sigma(i)}}-x_{{\sigma(i-1)}})\,{m_tex}'
        else:
            rhs = rf'\max_{{1\le i\le n}}\min\left(x_{{\sigma(i)}},\,{m_tex}\right)'
        if measure in ('weighted', 'owa', 'k_fuzzy') and str(weights).strip():
            tokens = [tok for tok in re.split(r'[\s,;]+', str(weights).strip()) if tok]
            if tokens:
                rhs += rf', \quad w = ({", ".join(tokens)})'
    else:
        formula = AGGREGATION_FORMULAS.get(name, '')
        if not formula:
            return ''
        lhs, rhs = rf'A_{{{sub}}}(x)', formula
    return rf'\({lhs} = {rhs}\)'


def format_method_label(item):
    """Build a concise library-style label for a configured aggregation item."""
    name = item['name']
    params = item.get('params', {})
    parts = [f'{k}={params[k]}' for k in PARAMETERS.get(name, ()) if k in params]
    if name in ('A_ch', 'A_sg'):
        measure = item.get('measure', 'uniform')
        if measure == 'exponential_cardinality':
            parts.append(f"measure=exp(q={params.get('measure_q', 1)})")
        else:
            parts.append(f'measure={measure}')
        if measure in ('weighted', 'owa', 'k_fuzzy') and str(item.get('weights', '')).strip():
            tokens = [tok for tok in re.split(r'[\s,;]+', str(item['weights']).strip()) if tok]
            if tokens:
                parts.append(f"w=[{','.join(tokens)}]")
    return f"{name}({', '.join(parts)})" if parts else f'{name}()'

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_ESTIMATED_RAM_BYTES = 128 * 1024 * 1024
MAX_COMPLEXITY_UNITS = 2_000_000
MAX_FULL_VIEW_ROWS = 2_000
METHOD_COMPLEXITY_WEIGHTS = {
    'A_ex2': 200,
    'A_ex3': 200,
    'A_lo': 5,
    'A_ch': 4,
    'A_sg': 4,
}


def validate_csv_upload(datapath, max_bytes=MAX_FILE_BYTES, max_rows=MAX_ROWS):
    """Validate uploaded CSV byte size and line count before loading into pandas."""
    path = Path(datapath)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f'The uploaded file ({size / (1024 * 1024):.1f} MB) exceeds the '
            f'{max_bytes / (1024 * 1024):.0f} MB limit.'
        )
    lines = 0
    last_byte = b'\n'
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            lines += chunk.count(b'\n')
            last_byte = chunk[-1:]
            if lines > max_rows + 1:
                raise ValueError(
                    f'The uploaded file has too many rows (limit: {max_rows:,} rows).'
                )
    if size > 0 and last_byte != b'\n':
        lines += 1
    if lines > max_rows + 1:
        raise ValueError(
            f'The uploaded file has too many rows (limit: {max_rows:,} rows).'
        )


def _is_numeric_cell(value, decimal='.'):
    text = value.strip()
    if not text:
        return False
    if decimal != '.':
        if '.' in text:
            return False
        text = text.replace(decimal, '.', 1)
    try:
        float(text)
        return True
    except ValueError:
        return False


def detect_csv_has_header(datapath, sep=',', decimal='.', sample_rows=20):
    """Heuristically detect whether the first row of a CSV file is a header."""
    rows = []
    with Path(datapath).open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
        reader = csv.reader(handle, delimiter=sep)
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append(row)
                if len(rows) >= sample_rows:
                    break
    if not rows:
        return True
    first = rows[0]
    if len(rows) == 1:
        return not all(_is_numeric_cell(cell, decimal) for cell in first if cell.strip())
    rest = rows[1:]
    for col_idx, cell in enumerate(first):
        if not cell.strip():
            continue
        if not _is_numeric_cell(cell, decimal):
            if any(col_idx < len(r) and _is_numeric_cell(r[col_idx], decimal) for r in rest):
                return True
    if any(_is_numeric_cell(cell, decimal) for cell in first if cell.strip()):
        return False
    return True


def estimate_workload(frame, columns, extra='', name='A_ar', kind='uniform',
                      max_rows=MAX_ROWS, max_ram_bytes=MAX_ESTIMATED_RAM_BYTES,
                      max_complexity=MAX_COMPLEXITY_UNITS):
    """Estimate memory and computational cost before running an aggregation."""
    extra_rows = sum(1 for line in extra.splitlines() if line.strip())
    rows = len(frame) + extra_rows
    cols = len(columns)
    if rows > max_rows:
        raise ValueError(
            f'Too many rows to process ({rows:,} rows; limit is {max_rows:,}).'
        )
    frame_bytes = int(frame.memory_usage(index=True, deep=False).sum()) if not frame.empty else 0
    table_copy_bytes = frame_bytes + extra_rows * max(1, len(frame.columns)) * 8
    values_bytes = rows * max(1, cols) * 8
    result_bytes = rows * 8
    measure_bytes = (
        (1 << min(cols, 20)) * 64
        if name in ('A_ch', 'A_sg') and kind in ('weighted', 'owa', 'k_fuzzy')
        else 0
    )
    estimated_ram_bytes = table_copy_bytes + values_bytes + result_bytes + measure_bytes
    if estimated_ram_bytes > max_ram_bytes:
        raise ValueError(
            f'Estimated memory usage ({estimated_ram_bytes / (1024 * 1024):.1f} MB) '
            f'exceeds the {max_ram_bytes / (1024 * 1024):.0f} MB server limit.'
        )
    weight = METHOD_COMPLEXITY_WEIGHTS.get(name, 1)
    if name in ('A_ch', 'A_sg') and kind in ('weighted', 'owa', 'k_fuzzy', 'exponential_cardinality'):
        weight = 10
    complexity = rows * max(1, cols) * weight
    if complexity > max_complexity:
        raise ValueError(
            f'The selected aggregation ({name}) on {rows:,} rows × {cols} columns '
            f'is too computationally expensive for interactive processing.'
        )
    return {
        'rows': rows,
        'cols': cols,
        'estimated_ram_bytes': estimated_ram_bytes,
        'complexity': complexity,
    }


def numbers(text):
    return [float(v) for v in re.split(r'[\s,;]+', text.strip()) if v]


def grid_view(frame, full=False, sort=None, count=5, max_full_rows=MAX_FULL_VIEW_ROWS):
    """Sort all real rows before adding the display-only head/tail gap."""
    table = frame.copy()
    label = 'Row'
    while label in table.columns:
        label += '_'
    table.insert(0, label, range(1, len(table) + 1))
    if sort and 0 <= sort['index'] < len(table.columns):
        table = table.sort_values(table.columns[sort['index']],
                                  ascending=sort['direction'] == 'asc',
                                  kind='stable', na_position='last')
    limit = max_full_rows if full else count * 2
    if len(table) <= limit:
        return table
    edge = max(1, max_full_rows // 2) if full else count
    gap = pd.DataFrame([['…'] * len(table.columns)], columns=table.columns)
    return pd.concat([table.head(edge), gap, table.tail(edge)], ignore_index=True)


def header_weight_vector(kind, columns, values):
    """Collect column weights, or OWA rank weights, in aggregation order."""
    result = []
    for rank, column in enumerate(columns, 1):
        key = str(rank) if kind == 'owa' else column
        value = values.get(key, '')
        if not str(value).strip():
            label = f'rank {rank} (largest value first)' if kind == 'owa' else f'column {column}'
            raise ValueError(f'Enter a weight for {label}. Provide {len(columns)} weights in total.')
        number = float(value)
        if not np.isfinite(number) or not 0 <= number <= 1:
            raise ValueError('Each weight must be a finite number in [0, 1].')
        result.append(str(number))
    return ' '.join(result)


def prepare_table(frame, columns, extra):
    if not columns:
        raise ValueError('Select at least one column.')
    table = frame.copy()
    rows = [numbers(line) for line in extra.splitlines() if line.strip()]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError('Each additional vector must contain one value for each selected column.')
    if rows:
        table = pd.concat([table, pd.DataFrame(rows, columns=columns)], ignore_index=True)
    if table.empty:
        raise ValueError('There are no rows to aggregate.')
    values = table.loc[:, columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('The selected columns contain missing or infinite values.')
    return table, values


def configuration(name, parameters, kind, weights, n):
    if name not in AGGREGATIONS:
        raise ValueError('Unknown aggregation method.')
    kwargs = {key: float(parameters[key]) for key in PARAMETERS.get(name, ())}
    if not all(np.isfinite(value) for value in kwargs.values()):
        raise ValueError('Parameters must be finite numbers.')
    if name in ('A_pw', 'A_ex') and kwargs['r'] == 0:
        raise ValueError('Parameter r must be nonzero.')
    if name in ('A_amn', 'A_amx') and not 0 <= kwargs['p'] <= 1:
        raise ValueError('Parameter p must be in [0, 1].')
    if name == 'A_oln':
        if kwargs['p'] < 1 or not kwargs['p'].is_integer():
            raise ValueError('The number of values to discard, p, must be a positive integer.')
        kwargs['p'] = int(kwargs['p'])
    if name in ('A_ex2', 'A_ex3'):
        if kwargs['p'] <= 0 or kwargs['p'] == 1:
            raise ValueError('Base p must be positive and different from 1.')
        if name == 'A_ex2' and (kwargs['q'] <= 0 or kwargs['q'] == 1):
            raise ValueError('Base q must be positive and different from 1.')
        if name == 'A_ex3' and kwargs['q'] == 0:
            raise ValueError('Parameter q must be nonzero.')
    args = [f'{key}={value!r}' for key, value in kwargs.items()]
    has_measure = False
    if name in ('A_ch', 'A_sg'):
        has_measure = True
        if kind not in MEASURES:
            raise ValueError('Unknown measure.')
        if kind == 'exponential_cardinality':
            q = float(parameters.get('measure_q', 1))
            if n > 12:
                raise ValueError('The exponential cardinality measure supports up to 12 columns in this app.')
            kwargs['measure'] = aggregation.exponential_cardinality_measure(n, q)
            measure_expr = f'exponential_cardinality_measure({n!r}, {q!r})'
        elif kind in ('uniform', 'relative'):
            argument = n
        else:
            argument = numbers(weights)
            if len(argument) != n:
                raise ValueError('Enter one weight for each selected column, in the same order.')
            # These constructors enumerate all subsets; keep the local UI responsive.
            if n > 12:
                raise ValueError('The weighted, OWA, and k-fuzzy measures support up to 12 columns in this app.')
        if kind != 'exponential_cardinality':
            kwargs['measure'] = getattr(aggregation, kind + '_measure')(argument)
            measure_expr = f'{kind}_measure({argument!r})'
        args.append(f'measure={measure_expr}')
    expression = f"{name}({', '.join(args + ['axis=1'])})"
    return getattr(ndarray, name)(**kwargs, axis=1), has_measure, expression


def _run_operator(operator, values, on_progress=None, chunk_size=500, offset=0, grand_total=None):
    total = len(values)
    overall = total if grand_total is None else grand_total
    if on_progress is None and total <= chunk_size:
        with np.errstate(all='ignore'):
            result = np.asarray(operator(values))
    else:
        parts = []
        if on_progress is not None:
            on_progress(offset, overall)
        with np.errstate(all='ignore'):
            for start in range(0, total, chunk_size):
                chunk = values[start:start + chunk_size]
                parts.append(np.asarray(operator(chunk)))
                if on_progress is not None:
                    on_progress(min(offset + start + len(chunk), overall), overall)
        result = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if np.iscomplexobj(result) or not np.isfinite(result).all():
        raise ValueError('The data or parameters produced a result outside the domain of the aggregation method. The result must be a finite real number.')
    return result


def _build_source_header(frame, columns, extra, imports, separator, decimal, header, uploaded):
    if uploaded:
        source = f"df = pd.read_csv('input.csv', sep={separator!r}, decimal={decimal!r}, header={0 if header else None!r})\n"
        if not header:
            source += "df.columns = [f'x{i + 1}' for i in range(len(df.columns))]\n"
    else:
        imports += 'from io import StringIO\n'
        source = f'df = pd.read_csv(StringIO({frame.to_csv(index=False)!r}))\n'
    source += f'columns = {list(columns)!r}\n'
    rows = [numbers(line) for line in extra.splitlines() if line.strip()]
    if rows:
        source += f'extra = pd.DataFrame({rows!r}, columns=columns)\ndf = pd.concat([df, extra], ignore_index=True)\n'
    return imports + '\n' + source + (
        "x = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)\n"
        "if not np.isfinite(x).all():\n    raise ValueError('Missing or non-finite input')\n"
    )


def calculate(frame, columns, extra, name, parameters, kind='uniform', weights='',
              separator=',', decimal='.', header=True, uploaded=False,
              on_progress=None, chunk_size=500):
    estimate_workload(frame, columns, extra, name, kind)
    table, values = prepare_table(frame, columns, extra)
    operator, has_measure, expression = configuration(name, parameters, kind, weights, len(columns))
    result = _run_operator(operator, values, on_progress=on_progress, chunk_size=chunk_size)
    output_name = name
    while output_name in table.columns:
        output_name += '_result'
    table[output_name] = result
    imports = f'import numpy as np\nimport pandas as pd\nfrom aggregationslib.ndarray import {name}\n'
    if has_measure:
        imports += f'from aggregationslib.aggregation import {kind}_measure\n'
    code = _build_source_header(frame, columns, extra, imports, separator, decimal, header, uploaded) + (
        f'df[{output_name!r}] = {expression}(x)\n'
        + f"df.to_csv('results.csv', index=False, sep={separator!r}, decimal={decimal!r})\n"
    )
    return table, code


def calculate_many(frame, columns, methods, extra='', separator=',', decimal='.',
                   header=True, uploaded=False, on_progress=None, chunk_size=500):
    """Compute multiple aggregation configurations on the same selected columns."""
    if not methods:
        raise ValueError('Add at least one aggregation method.')
    total_ram = 0
    total_complexity = 0
    for item in methods:
        info = estimate_workload(frame, columns, extra, item['name'], item.get('measure', 'uniform'))
        total_ram += info['estimated_ram_bytes']
        total_complexity += info['complexity']
    if total_ram > MAX_ESTIMATED_RAM_BYTES:
        raise ValueError(
            f'Estimated memory usage ({total_ram / (1024 * 1024):.1f} MB) '
            f'exceeds the {MAX_ESTIMATED_RAM_BYTES / (1024 * 1024):.0f} MB server limit.'
        )
    if total_complexity > MAX_COMPLEXITY_UNITS:
        raise ValueError('The combined list of aggregations is too computationally expensive for interactive processing.')

    table, values = prepare_table(frame, columns, extra)
    grand_total = len(values) * len(methods)
    result_columns = []
    imported_names = []
    imported_measures = []
    body_lines = []

    for idx, item in enumerate(methods):
        name = item['name']
        params = item.get('params', {})
        kind = item.get('measure', 'uniform')
        weights = item.get('weights', '')
        operator, has_measure, expression = configuration(name, params, kind, weights, len(columns))
        result = _run_operator(
            operator, values, on_progress=on_progress, chunk_size=chunk_size,
            offset=idx * len(values), grand_total=grand_total,
        )
        col_base = format_method_label(item) if len(methods) > 1 else name
        output_name = col_base
        suffix = 2
        while output_name in table.columns:
            output_name = f'{col_base}_{suffix}' if len(methods) > 1 else f'{output_name}_result'
            suffix += 1
        table[output_name] = result
        result_columns.append(output_name)
        if name not in imported_names:
            imported_names.append(name)
        if has_measure and kind not in imported_measures:
            imported_measures.append(kind)
        body_lines.append(f'df[{output_name!r}] = {expression}(x)\n')

    imports = f"import numpy as np\nimport pandas as pd\nfrom aggregationslib.ndarray import {', '.join(imported_names)}\n"
    if imported_measures:
        measure_funcs = ', '.join(f'{m}_measure' for m in imported_measures)
        imports += f'from aggregationslib.aggregation import {measure_funcs}\n'
    code = _build_source_header(frame, columns, extra, imports, separator, decimal, header, uploaded) + ''.join(body_lines) + (
        f"df.to_csv('results.csv', index=False, sep={separator!r}, decimal={decimal!r})\n"
    )
    return table, code, result_columns


def calculate_many_per_class(class_frames, class_names, methods, separator=',', decimal='.',
                             header=True, on_progress=None, chunk_size=500):
    """Aggregate one CSV of model scores per class and predict via argmax."""
    if len(class_frames) < 2 or len(class_frames) != len(class_names):
        raise ValueError('Provide at least two class CSV files.')
    cleaned_names = [str(c).strip() for c in class_names]
    if any(not c for c in cleaned_names) or len(set(cleaned_names)) != len(cleaned_names):
        raise ValueError('Each class must have a unique, non-empty label.')
    if not methods:
        raise ValueError('Add at least one aggregation method.')

    row_counts = [len(df) for df in class_frames]
    if row_counts[0] == 0 or len(set(row_counts)) != 1:
        raise ValueError('All class CSV files must be non-empty and have the exact same number of rows.')

    per_class_values = []
    for cname, df in zip(cleaned_names, class_frames):
        cols = list(df.columns)
        if not cols:
            raise ValueError(f'Class {cname!r} CSV has no columns.')
        _, values = prepare_table(df, cols, '')
        per_class_values.append(values)

    total_ram = 0
    total_complexity = 0
    for item in methods:
        for df in class_frames:
            info = estimate_workload(df, list(df.columns), '', item['name'], item.get('measure', 'uniform'))
            total_ram += info['estimated_ram_bytes']
            total_complexity += info['complexity']
    if total_ram > MAX_ESTIMATED_RAM_BYTES:
        raise ValueError(
            f'Estimated memory usage ({total_ram / (1024 * 1024):.1f} MB) '
            f'exceeds the {MAX_ESTIMATED_RAM_BYTES / (1024 * 1024):.0f} MB server limit.'
        )
    if total_complexity > MAX_COMPLEXITY_UNITS:
        raise ValueError('The combined list of aggregations is too computationally expensive for interactive processing.')

    n_rows = row_counts[0]
    n_classes = len(cleaned_names)
    grand_total = n_rows * n_classes * len(methods)
    table = pd.DataFrame(index=range(n_rows))
    score_matrices = {}
    result_columns = []
    imported_names = []
    imported_measures = []
    body_lines = []
    done_offset = 0

    for item in methods:
        name = item['name']
        params = item.get('params', {})
        kind = item.get('measure', 'uniform')
        weights = item.get('weights', '')
        col_base = format_method_label(item) if len(methods) > 1 else name
        base_name = col_base
        suffix = 2
        while base_name in score_matrices:
            base_name = f'{col_base}_{suffix}'
            suffix += 1

        class_scores = []
        for c_idx, (cname, values) in enumerate(zip(cleaned_names, per_class_values)):
            operator, has_measure, expression = configuration(name, params, kind, weights, values.shape[1])
            scores = _run_operator(
                operator, values, on_progress=on_progress, chunk_size=chunk_size,
                offset=done_offset, grand_total=grand_total,
            )
            done_offset += n_rows
            class_scores.append(scores)
            table[f'{base_name} [{cname}]'] = scores
            if name not in imported_names:
                imported_names.append(name)
            if has_measure and kind not in imported_measures:
                imported_measures.append(kind)
            if c_idx == 0:
                body_lines.append(f'agg_{len(score_matrices) + 1} = {expression}\n')

        mat = np.column_stack(class_scores)
        score_matrices[base_name] = mat
        result_columns.append(base_name)
        pred_idx = np.argmax(mat, axis=1)
        table[f'{base_name} [pred]'] = [cleaned_names[i] for i in pred_idx]
        body_lines.append(
            f'scores_{len(score_matrices)} = np.column_stack([agg_{len(score_matrices)}(x_c) for x_c in class_arrays])\n'
            f"df[{f'{base_name} [pred]'!r}] = [classes[i] for i in np.argmax(scores_{len(score_matrices)}, axis=1)]\n"
        )

    imports = f"import numpy as np\nimport pandas as pd\nfrom aggregationslib.ndarray import {', '.join(imported_names)}\n"
    if imported_measures:
        measure_funcs = ', '.join(f'{m}_measure' for m in imported_measures)
        imports += f'from aggregationslib.aggregation import {measure_funcs}\n'
    file_list = [f'class_{i}.csv' for i in range(n_classes)]
    setup = (
        f'classes = {cleaned_names!r}\n'
        f'files = {file_list!r}\n'
        f"class_arrays = [pd.read_csv(f, sep={separator!r}, decimal={decimal!r}, header={0 if header else None!r}).to_numpy(dtype=float) for f in files]\n"
        'df = pd.DataFrame()\n'
    )
    code = imports + '\n' + setup + ''.join(body_lines) + (
        f"df.to_csv('results.csv', index=False, sep={separator!r}, decimal={decimal!r})\n"
    )
    return table, code, result_columns, score_matrices
