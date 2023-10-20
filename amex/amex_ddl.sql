-- DDL for lec example week 2
create table amex_features(
    transaction_ID  integer primary key,
    customer_ID     text,
    S_2             timestamp,
    target          integer,
    D_63            text,
    S_6             decimal(9, 6),
    R_16            decimal(9, 6),
    B_10            decimal(9, 6),
    R_1             decimal(9, 6),
    D_127           decimal(9, 6),
    B_36            decimal(9, 6),
    D_65            decimal(9, 6),
    R_2             decimal(9, 6),
    B_24            decimal(9, 6),
    B_4             decimal(9, 6),
    D_92            decimal(9, 6)
);
