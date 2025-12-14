"""
Visualization module for emotional contagion analysis.

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_emotion_barplot(df, col = 'emotion_child', title = "Dominant Emotion of Child Comments", xlabel = 'Emotion', ylabel = 'Count', path = None): 
    """
    Creates a barplot displaying counts of emotions
    Input: dataframe containing emotion data, column name for categorical analysis, title, xlabel, and ylabel for plot, 
        and a path to save the file if desired
    Return: finished plot
    
    """
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize = (8, 5))
    bars = ax.bar(counts.index, counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') 
    if path:
        fig.savefig(path, bbox_inches='tight')
    
    return fig

def plot_valence_hist(df, col = 'valence_child', title = "Distribution of Valence Scores Among Child Comments", xlabel = 'Valence Score', ylabel = 'Count', path = None, bin = 20):
    """
    Creates histogram of a valence frequencies
    Input: dataframe containing valence scores, title, xlabel, and ylabel for plot, bin number, and optional path
        for saving figure
    Return: finished plot
    """
    fig, ax = plt.subplots()
    plt.hist(df[col], bins=bin)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig

def plot_emotion_corr_heatmap(df, cols = ['valence_parent', 'valence_child'], title = "Correlation of Parent and Child Valence", path = None):
    """
    Creates heatmap of emotion correlations
    Input: dataframe containing different emotion probabilities, column of emotion probabilities, title for plot,
        path if saving figure
    Return: finished plot
    """
    fig, ax = plt.subplots()
    sns.heatmap(df[cols].corr(), annot=True, cmap = 'crest', ax= ax)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig

def plot_average_emotion_probs(df, cols = 'emotion_parent', title = "Barplot of  Average Emotion Probabilies", xlabel = 'Emotion', ylabel = 'Average Probability', path = None):
    """
    Creates barplot of average probabilities of comments
    Input: dataframe containing different emotion probabilities, columns of emotion probabilities, title, xlabel, and ylabel for plot,
        and path if saving figure
    Return: finished plot
    """
    fig, ax = plt.subplots()
    df[cols].mean().plot(kind= 'bar', ax = ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig

def plot_parent_child_valence_scatter(df, parent_valence = 'valence_parent', child_valence = 'valence_child', title = "Correlation Between Child Valence and Parent Valence", 
                                          xlabel = 'Parent Valence', ylabel = 'Child Valence', path = None):
    """
    Creates scatterplot of parent valence vs child valence
    Input: dataframe containing different parent valence and child valence, column of emotion probabilities, title, xlabel, and ylabel for plot,
        and path if saving figure
    Return: finished plot
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x = parent_valence, y = child_valence, data = df, ax = ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig

def plot_depth_valence_correlation(df, depth_col = 'delta_depth', child_valence = 'valence_child', title = "Correlation Between Child Valence and Depth from Parent Comment", 
                                          xlabel = 'Depth from Parent Comment', ylabel = 'Child Valence', path = None):
    """
    Creates scatterplot of parent emotion probability vs child valence
    Input: dataframe containing different emotion probabilities and child valence, column of emotion probabilities, title, xlabel, and ylabel for plot,
        and path if saving figure
    Return: finished plot
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x = depth_col, y = child_valence, data = df, ax = ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig

def build_transition_matrix(df):
    """
    Create parent→child emotion transition counts and probabilities.
    """
    # Extract the two categorical emotion columns
    parent_em = df["emotion_parent"]
    child_em = df["emotion_child"]

    # Count transitions
    transition_counts = pd.crosstab(parent_em, child_em)

    # Probability matrix = row-normalized counts
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    return transition_counts, transition_probs

def plot_transition_heatmap(probs, title = "Emotion Transition Probability Heatmap", xlabel = "Child Emotion", ylabel = "Parent Emotion", path = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(probs, annot=False, cmap="mako", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    print("Testing plot functions on default data:")
    df = pd.read_parquet('../../../data/contagion_ready.parquet')
    _ , probs = build_transition_matrix(df)
    plot_emotion_barplot(df, path ='../../../figures/emotion_barplot.png')
    plot_valence_hist(df, path ='../../../figures/valence_histogram.png')
    #plot_average_emotion_probs(df, path ='../../../figures/emotion_probs.png')
    plot_parent_child_valence_scatter(df, path ='../../../figures/valence_scatter.png')
    plot_emotion_corr_heatmap(df, path ='../../../figures/valence_heatmap.png')
    plot_depth_valence_correlation(df, path ='../../../figures/valence_depth_scatter.png')
    plot_transition_heatmap(probs, path = '../../../figures/emotion_transitions.png')
    
    print("Test plots saved to /figures")