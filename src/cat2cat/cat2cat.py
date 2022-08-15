def cat2cat(
    data={
        "old": None,
        "new": None,
        "cat_var": None,
        "cat_var_old": None,
        "cat_var_new": None,
        "id_var": None,
        "time_var": None,
        "multiplier_var": None,
        "freqs_df": None,
    },
    mappings={"trans": None, "direction": None},
    ml={"data": None, "cat_var": None, "method": None, "features": None, "args": None},
):
    """
    data args
    \itemize{
    \item{"old"}{ data.frame older time point in a panel}
    \item{"new"} { data.frame more recent time point in a panel}
    \item{"cat_var"}{ character name of the categorical variable.}
    \item{"cat_var_old"}{ optional character name of the categorical variable in the older time point. Default `cat_var`.}
    \item{"cat_var_new"}{ optional character name of the categorical variable in the newer time point. Default `cat_var`.}
    \item{"time_var"}{ character name of the time variable.}
    \item{"id_var"}{ optional character name of the unique identifier variable - if this is specified then for subjects observe in both periods the direct mapping is applied.}
    \item{"multiplier_var"}{ optional character name of the multiplier variable - number of replication needed to reproduce the population}
    \item{"freqs_df"}{ optional - data.frame with 2 columns where first one is category name and second counts.
    It is optional nevertheless will be very often needed, as give more control.
    It will be used to assess the probabilities. The multiplier variable is omit so sb has to apply it in this table.}
    }
    mappings args
    \itemize{
    \item{"trans"}{ data.frame with 2 columns - transition table - all categories for cat_var in old and new datasets have to be included.
    First column contains an old encoding and second a new one.
    The transition table should to have a candidate for each category from the targeted for an update period.
    }
    \item{"direction"}{ character direction - "backward" or "forward"}
    }
    optional ml args
    \itemize{
    \item{"data"}{ data.frame - dataset with features and the `cat_var`.}
    \item{"cat_var"}{ character - the dependent variable name.}
    \item{"method"}{ character vector - one or a few from "knn", "rf" and "lda" methods - "knn" k-NearestNeighbors, "lda" Linear Discrimination Analysis, "rf" Random Forest }
    \item{"features"}{ character vector of features names where all have to be numeric or logical}
    \item{"args"}{ optional - list parameters: knn: k ; rf: ntree  }
    }
    @return named list with 2 fields old an new - 2 data.frames.
    There will be added additional columns like index_c2c, g_new_c2c, wei_freq_c2c, rep_c2c, wei_(ml method name)_c2c.
    Additional columns will be informative only for a one data.frame as we always make a changes to one direction.
    """
    return None
