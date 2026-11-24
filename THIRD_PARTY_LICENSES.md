# Third-party licenses

MATwin bundles the source code listed first. It does not redistribute the packages listed after that: `environment.yml` installs them, and each keeps its own license.

## Bundled source code

| Package | Location | License | Copyright | URL |
|---|---|---|---|---|
| isuelogit (modified) | `src/isuelogit` | MIT | Copyright (c) 2022 Pablo Guarda | https://github.com/pabloguarda/isuelogit |
| pesuelogit (modified) | `src/pesuelogit` | MIT | Copyright (c) 2023 Pablo Guarda | https://github.com/pabloguarda/pesuelogit |
| MaTE (modified) | `src/mate` | MIT | Copyright (c) 2024 Pablo Guarda | https://github.com/pabloguarda/mate |
| MaTE architecture figure | `docs/images/mate-general-architecture.png` | MIT | Copyright (c) 2024 Pablo Guarda | https://github.com/pabloguarda/mate |

Each package's MIT license text, with its copyright line, is in `src/<package>/LICENSE`.

## Direct dependencies (environment.yml)

| Package | Version | License | URL |
|---|---|---|---|
| python | 3.8 (3.8.18 tested) | PSF-2.0 | https://www.python.org |
| pip | not pinned (24.0 tested) | MIT | https://pip.pypa.io |
| numpy | 1.24.4 | BSD-3-Clause | https://numpy.org |
| scipy | 1.10.1 | BSD-3-Clause | https://scipy.org |
| pandas | 2.0.3 | BSD-3-Clause | https://pandas.pydata.org |
| matplotlib-base | 3.7.3 | Matplotlib License (PSF-based) | https://matplotlib.org |
| seaborn | 0.13.2 | BSD-3-Clause | https://seaborn.pydata.org |
| scikit-learn | 1.3.2 | BSD-3-Clause | https://scikit-learn.org |
| statsmodels | 0.14.1 | BSD-3-Clause | https://www.statsmodels.org |
| pykrige | 1.7.1 | BSD-3-Clause | https://github.com/GeoStat-Framework/PyKrige |
| networkx | 3.1 | BSD-3-Clause | https://networkx.org |
| pytables | 3.7.0 | BSD-3-Clause | https://www.pytables.org |
| numdifftools | 0.9.41 | BSD-3-Clause | https://github.com/pbrod/numdifftools |
| algopy | 0.5.7 | BSD | https://github.com/b45ch1/algopy |
| sortedcontainers | 2.4.0 | Apache-2.0 | https://grantjenks.com/docs/sortedcontainers/ |
| psutil | 5.9.8 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| requests | 2.31.0 | Apache-2.0 | https://requests.readthedocs.io |
| beautifulsoup4 | 4.12.3 | MIT | https://www.crummy.com/software/BeautifulSoup/ |
| shapely | 1.8.5 | BSD-3-Clause | https://github.com/shapely/shapely |
| geopandas | 0.13.2 | BSD-3-Clause | https://geopandas.org |
| osmnx | 1.2.2 | MIT | https://github.com/gboeing/osmnx |
| contextily | 1.5.0 | BSD-3-Clause | https://github.com/geopandas/contextily |
| folium | 0.15.1 | MIT | https://github.com/python-visualization/folium |
| branca | 0.7.1 | MIT | https://github.com/python-visualization/branca |
| jupyter | 1.0.0 | BSD-3-Clause | https://jupyter.org |
| tensorflow-cpu | 2.11.0 | Apache-2.0 | https://www.tensorflow.org |
| census | 0.8.21 | BSD-3-Clause | https://github.com/datamade/census |
| openmatrix | 0.3.5.0 | Apache-2.0 | https://github.com/osPlanning/omx-python |
| papermill | 2.6.0 | BSD-3-Clause | https://github.com/nteract/papermill |
| pytest | 8.3.5 | MIT | https://github.com/pytest-dev/pytest |

The data sources used by the examples and their terms are listed in the README under "Data sources and attribution".
