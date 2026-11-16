"""Data preparation shared by the Shiny app and its generated examples."""
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
    'A_lm': 'Lehmer mean', 'A_amn': 'Arithmetic minimum mean',
    'A_amx': 'Arithmetic maximum mean', 'A_md': 'Median',
    'A_ol': 'Olympic mean', 'A_oln': 'Generalized Olympic mean',
    'A_lo': 'Logarithmic mean', 'A_ch': 'Choquet integral',
    'A_sg': 'Sugeno integral',
}
PARAMETERS = {
    'A_pw': ('r',), 'A_ex': ('r',), 'A_lm': ('r',),
    'A_ex2': ('p', 'q'), 'A_ex3': ('p', 'q'),
    'A_amn': ('p',), 'A_amx': ('p',), 'A_oln': ('p',),
}
MEASURES = {'uniform': 'Uniform', 'relative': 'Relative',
            'exponential_cardinality': 'Exponential cardinality',
            'weighted': 'Weighted', 'owa': 'OWA', 'k_fuzzy': 'k-fuzzy (λ)'}

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
    prefix = ''
    if name in ('A_ch', 'A_sg'):
        if kind not in MEASURES:
            raise ValueError('Unknown measure.')
        if kind == 'exponential_cardinality':
            q = float(parameters.get('measure_q', 1))
            if n > 12:
                raise ValueError('The exponential cardinality measure supports up to 12 columns in this app.')
            kwargs['measure'] = aggregation.exponential_cardinality_measure(n, q)
            prefix = f'measure = exponential_cardinality_measure({n!r}, {q!r})\n'
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
            prefix = f'measure = {kind}_measure({argument!r})\n'
        args.append('measure=measure')
    expression = f"{name}({', '.join(args + ['axis=1'])})"
    return getattr(ndarray, name)(**kwargs, axis=1), prefix, expression


def calculate(frame, columns, extra, name, parameters, kind='uniform', weights='',
              separator=',', decimal='.', header=True, uploaded=False,
              on_progress=None, chunk_size=500):
    estimate_workload(frame, columns, extra, name, kind)
    table, values = prepare_table(frame, columns, extra)
    operator, prefix, expression = configuration(name, parameters, kind, weights, len(columns))
    total = len(values)
    if on_progress is None and total <= chunk_size:
        with np.errstate(all='ignore'):
            result = np.asarray(operator(values))
    else:
        parts = []
        if on_progress is not None:
            on_progress(0, total)
        with np.errstate(all='ignore'):
            for start in range(0, total, chunk_size):
                chunk = values[start:start + chunk_size]
                parts.append(np.asarray(operator(chunk)))
                if on_progress is not None:
                    on_progress(min(start + len(chunk), total), total)
        result = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if np.iscomplexobj(result) or not np.isfinite(result).all():
        raise ValueError('The data or parameters produced a result outside the domain of the aggregation method. The result must be a finite real number.')
    output_name = name
    while output_name in table.columns:
        output_name += '_result'
    table[output_name] = result
    imports = f'import numpy as np\nimport pandas as pd\nfrom aggregationslib.ndarray import {name}\n'
    if prefix:
        imports += f'from aggregationslib.aggregation import {kind}_measure\n'
    if uploaded:
        source = f"df = pd.read_csv('input.csv', sep={separator!r}, decimal={decimal!r}, header={0 if header else None!r})\n"
        if not header:
            source += "df.columns = [f'x{i + 1}' for i in range(len(df.columns))]\n"
    else:
        # Serialize the original table, including identifier columns.
        imports += 'from io import StringIO\n'
        source = f'df = pd.read_csv(StringIO({frame.to_csv(index=False)!r}))\n'
    source += f'columns = {list(columns)!r}\n'
    rows = [numbers(line) for line in extra.splitlines() if line.strip()]
    if rows:
        source += f'extra = pd.DataFrame({rows!r}, columns=columns)\ndf = pd.concat([df, extra], ignore_index=True)\n'
    code = imports + '\n' + source + (
        "x = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)\n"
        "if not np.isfinite(x).all():\n    raise ValueError('Missing or non-finite input')\n"
        + prefix + f'aggregate = {expression}\n'
        + f'df[{output_name!r}] = aggregate(x)\n'
        + f"df.to_csv('results.csv', index=False, sep={separator!r}, decimal={decimal!r})\n"
    )
    return table, code
