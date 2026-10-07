import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# update/add code below ...

TITANIC_URL = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv'


def to_snake_case(name):
    name = name.replace(' ', '_')
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    return name.lower()


def load_titanic():
    df = pd.read_csv(TITANIC_URL)
    df.columns = [to_snake_case(col) for col in df.columns]
    return df


def survival_demographics():
    df = load_titanic()
    df = df.dropna(subset=['age'])

    bins = [0, 12, 19, 59, np.inf]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    df['age_group'] = pd.Categorical(df['age_group'], categories=labels, ordered=True)

    result = (
        df.groupby(['pclass', 'sex', 'age_group'], observed=False)
        .agg(n_passengers=('survived', 'size'), n_survivors=('survived', 'sum'))
        .reset_index()
    )
    result['survival_rate'] = result['n_survivors'] / result['n_passengers']
    result = result.sort_values(['pclass', 'sex', 'age_group']).reset_index(drop=True)

    return result


def visualize_demographic():
    # Question: How does survival rate vary across passenger class, sex, and age
    # group, and which of these three factors seems to matter most?
    data = survival_demographics().copy()
    data['class_label'] = data['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    data['sex_label'] = data['sex'].str.title()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Approach 1: Survival Rate by Class & Sex',
            'Approach 2: Survival Rate Heatmap (Sex + Age Group vs Class)',
            'Approach 3: Survival Rate Trend by Age Group',
            'Approach 4: Group Size (bubble) vs Survival Rate (color)',
        ),
    )

    # Approach 1: grouped bar chart, survival rate by class & sex
    # (aggregated across age groups)
    agg_class_sex = (
        data.groupby(['class_label', 'sex_label'])
        .apply(lambda g: g['n_survivors'].sum() / g['n_passengers'].sum(), include_groups=False)
        .reset_index(name='survival_rate')
    )
    colors = {'Male': '#4C78A8', 'Female': '#F58518'}
    for sex_label, subset in agg_class_sex.groupby('sex_label'):
        fig.add_trace(
            go.Bar(
                x=subset['class_label'], y=subset['survival_rate'],
                name=sex_label, marker_color=colors[sex_label],
                legendgroup=sex_label,
            ),
            row=1, col=1,
        )

    # Approach 2: heatmap of survival rate, (sex + age group) x class
    data['group_label'] = data['sex_label'] + ' - ' + data['age_group'].astype(str)
    heatmap_data = data.pivot_table(index='group_label', columns='class_label', values='survival_rate')
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
            colorscale='RdYlGn', zmin=0, zmax=1, showscale=True,
            colorbar=dict(title='Survival Rate', len=0.4, y=0.8),
        ),
        row=1, col=2,
    )

    # Approach 3: line chart, survival rate trend across age groups,
    # one line per class/sex combination
    for (class_label, sex_label), grp in data.groupby(['class_label', 'sex_label']):
        grp = grp.sort_values('age_group')
        fig.add_trace(
            go.Scatter(
                x=grp['age_group'].astype(str), y=grp['survival_rate'],
                mode='lines+markers', name=f'{class_label} - {sex_label}',
                legendgroup=f'{class_label}-{sex_label}', showlegend=False,
            ),
            row=2, col=1,
        )

    # Approach 4: bubble chart, bubble size = group size, color = survival rate
    fig.add_trace(
        go.Scatter(
            x=data['class_label'], y=data['group_label'],
            mode='markers',
            marker=dict(
                size=data['n_passengers'], sizemode='area',
                sizeref=2. * data['n_passengers'].max() / (40. ** 2), sizemin=4,
                color=data['survival_rate'], colorscale='RdYlGn', cmin=0, cmax=1,
                showscale=False,
            ),
            showlegend=False,
            text=data['n_passengers'],
            hovertemplate='n_passengers=%{text}<br>survival_rate=%{marker.color:.2f}<extra></extra>',
        ),
        row=2, col=2,
    )

    fig.update_yaxes(range=[0, 1], title_text='Survival Rate', row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text='Survival Rate', row=2, col=1)
    fig.update_xaxes(title_text='Age Group', row=2, col=1)
    fig.update_layout(
        height=800, width=1000,
        title_text='Titanic Survival Demographics: Four Views of Class, Sex, and Age Group',
        barmode='group',
    )

    return fig


def family_groups():
    df = load_titanic()
    df['family_size'] = df['sib_sp'] + df['parch'] + 1

    result = (
        df.groupby(['pclass', 'family_size'])
        .agg(n_passengers=('fare', 'size'), avg_fare=('fare', 'mean'),
             min_fare=('fare', 'min'), max_fare=('fare', 'max'))
        .reset_index()
    )
    result = result.sort_values(['pclass', 'family_size']).reset_index(drop=True)

    return result


def last_names():
    df = load_titanic()
    last_name = df['name'].str.split(',').str[0].str.strip()
    return last_name.value_counts()


def visualize_families():
    # Question: Does average ticket fare rise with family size, and does the
    # relationship (and the spread between min/max fare) differ across classes?
    data = family_groups()
    class_labels = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
    colors = {1: '#4C78A8', 2: '#F58518', 3: '#54A24B'}

    fig = go.Figure()
    for pclass, grp in data.groupby('pclass'):
        grp = grp.sort_values('family_size')
        fig.add_trace(
            go.Scatter(
                x=grp['family_size'], y=grp['avg_fare'],
                mode='lines+markers',
                name=class_labels[pclass],
                marker=dict(
                    size=grp['n_passengers'], sizemode='area',
                    sizeref=2. * data['n_passengers'].max() / (40. ** 2), sizemin=4,
                    color=colors[pclass],
                ),
                line=dict(color=colors[pclass]),
                error_y=dict(
                    type='data', symmetric=False,
                    array=grp['max_fare'] - grp['avg_fare'],
                    arrayminus=grp['avg_fare'] - grp['min_fare'],
                    color=colors[pclass], thickness=1, width=2,
                ),
                hovertemplate='Family size=%{x}<br>Avg fare=$%{y:.2f}<br>n_passengers=%{marker.size}<extra></extra>',
            )
        )

    fig.update_layout(
        title='Average Ticket Fare by Family Size and Passenger Class<br><sup>Marker size = group size, error bars = min-max fare range</sup>',
        xaxis_title='Family Size', yaxis_title='Average Fare ($)',
        height=600, width=900,
    )

    return fig