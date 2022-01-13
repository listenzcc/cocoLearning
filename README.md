# Learn Coco Dataset

Learn Coco Dateset with its annotations.

## Prior Work

- Download the annotations of [Coco](https://cocodataset.org/ "Coco") dataset;
- Put them into the folder of `annotations`;
- The `overlook.py` script will automatically count the histogram of objects' bbox in the annotations.
- The results will be saved into the folder of `largeFiles`.

## Occlude Counts

You will get images like these

- Count heat map of Occlude

    ![Occlude Counts](./largeFiles/Occlude%20Count%20Heat%20Map.html.png)

- Histogram of categories

    ![Histogram of Categories](./largeFiles/appliance-toaster.html.png)

- Box graph of objects' area

    ![Box Graph](./largeFiles/Area%20-%20BoxGraph.html.png)