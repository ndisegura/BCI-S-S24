## README for BNCI Horizon 2020's P300 dataset

Written 8/18/21 by DJ.

Dataset, description, and (Guger, 2009) paper downloaded from http://bnci-horizon-2020.eu/database/data-sets
"Number 12: Visual P300 speller (003-2015)"

**Notes:**
- Rows of the data matrices are slightly different than described in the BNCI [description.pdf](description.pdf) file:
 - Row 0 is the time of each sample in seconds.
 - Rows 1-8 are the EEG data in unknown units (we will assume uV). The channels are described in [Guger2009.pdf](Guger2009.pdf), Figure 1 (Fz, Cz, P3, Pz, P4, PO7, OZ, PO8), but the order is unknown.
 - Row 9 is the current event type. 1=column 1, 2=column 2, ... 6=column 6, 7=row 1, 8=row 2, ... 12=row 6.
 - Row 10 is 1 when the current row/col flashed includes the target letter and 0 otherwise.
 - Letters are arranged in rows and columns as seen in [Guger2009.pdf](Guger2009.pdf), Figure 1.
- Subjects s1 and s2 have triggers up to 36, which suggests they're flashing single letters and not rows/cols as described in the BNCI description.pdf file.
- Subjects s3-10 have triggers up to 12 as described in the BNCI description.pdf file.
- The EEG data appears to be the raw, unprocessed data.
