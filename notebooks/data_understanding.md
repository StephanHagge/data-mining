# Data Understanding

## Does the data describe the process?

The data describe the process as stated. Additionally, required data are added or calculated.
For example a column for an overall rating was added. That value can now be used for further analyzing.

## Should further data be integrated?

For now no further data should not be integrated, except the data we calculated. Doing so would just lead to complexity.
But we could add additional attributes via [this](https://developers.google.com/youtube/v3/docs/videos/list)
API in the case it seems necessary.

## Are there (input) error within the data?

No errors have been detected so far.

## How should data be organized for data exploration and model creation?

As described, we only use the dataset related to Germany. It is organized as Table, containing video data.
The category in clear text is stored in a seperate JSON-File. In the dataset only the category_id is included.