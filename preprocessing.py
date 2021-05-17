# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import pandas as pd


# %%
df = pd.read_excel('dataset.xlsx')


# %%
df.head()


# %%
lc = df['Lyric']

# %% [markdown]
# ## Exploring Lyrics Column
# %% [markdown]
# #### How is the song length distributed

# %%
lc.str.len().describe()

# %% [markdown]
# #### Maximum song length in the 99th quantile

# %%
lc.str.len().quantile(0.99)

# %% [markdown]
# #### Plotting the 95th quantile

# %%
lc[lc.str.len() < 2695].str.len().hist(figsize=(20,5), bins=200)


# %%
df_cleaned = df[(lc.str.len() > 30) & (lc.str.len() < 2695)]

# %% [markdown]
# #### Duplicated values

# %%
df_cleaned = df_cleaned.drop_duplicates(subset='Lyric')

# %% [markdown]
# #### Removing songs with chords, i.e '------'

# %%
df_cleaned = df_cleaned[~lc.str.contains('------')]


# %%
df_cleaned


# %%
cols = ['Lyric', 'SName', 'artist']
df_cleaned[cols].to_csv('lyrics_artist_sname.csv', index=False)


