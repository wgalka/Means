"""Data preparation shared by the Shiny app and its generated examples."""
from io import StringIO
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


def numbers(text):
    return [float(v) for v in re.split(r'[\s,;]+', text.strip()) if v]


def grid_view(frame, full=False, sort=None, count=5):
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
    if full or len(table) <= count * 2:
        return table
    gap = pd.DataFrame([['…'] * len(table.columns)], columns=table.columns)
    return pd.concat([table.head(count), gap, table.tail(count)], ignore_index=True)


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
              separator=',', decimal='.', header=True, uploaded=False):
    table, values = prepare_table(frame, columns, extra)
    operator, prefix, expression = configuration(name, parameters, kind, weights, len(columns))
    with np.errstate(all='ignore'):
        result = np.asarray(operator(values))
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
