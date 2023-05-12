# SemRepFact
A large dataset of predications and their certainty/polarity values extracted from SemMedDB.


## Creation

1. Obtain a large number of predications in brat format from SemMedDB

```
python scripts/query_predications.py /path/to/semmed40.db /path/to/SemanticTypes_2018AB_revised.txt ann_outdir/ --max-predications N
```

Where `N` is your chosen number of predications. 
`SemanticTypes_2018AB_revised.txt` can be found in the `assets` directory.

The result will be N*3 files in `ann_outdir`, where for each PMID there are

 * PMID.ann: The brat annotations
 * PMID.txt: The raw abstract text
 * PMID.json: The sentence-segmented abstract

These files can be read in using `pybrat` like so.

```python
from pybrat import BratAnnotations, BratText

anns = BratAnnotations.from_file("ann_outdir/PMID.ann")
anntxt = BratText.from_files(text="ann_outdir/PMID.txt", sentences="ann_outdir/PMID.json")
```

This step takes about 1 minute for 100k predications.


2. Filter out incorrect predications using an ML model


```
python scripts/filter_predications.py ann_outdir/ assets/filtering/covid_clf/checkpoints/best_model.pth filtered_outdir/
```

This script symlinks the .txt and .json files from `ann_outdir` into `filtered_outdir`.


3. Run CoreNLP on the filtered text files, as this information is required for running SemRep Fact. This command takes the longest to run, at ~1.5 hours per 2500 input files.

```
cd Bio-SCoRes
bash bin/CoreNLP filtered_outdir/ corenlp_outdir/
```

CoreNLP annotations will be saved as XML files in `corenlp_outdir`.

There is also a parallelized version (using GNU parallel) that can be used like so

```
bash bin/CoreNLP_parallel N filtered_outdir/ corenlp_outdir/
```

where `N` is an integer specifying the number of jobs.


3. Convert brat annotations to XML and merge them into the CoreNLP XML files.

```
python scripts/merge_semrep_corenlp.py filtered_outdir/ corenlp_outdir/ xml_output_dir/
```

This script will add XML files to `xml_output_dir`.


The above script creates XML elements for Spans and Events like so.

```python
import xml.etree.ElementTree as ET

anns = BratAnnotations.from_file("filtered_outdir/PMID.txt")

span = anns.spans[0]
span_attribs = {"xml:space": "preserve",
							  "id": span.id,
								"type": span.type,
								"charOffset": f"{span.start_index}-{span.end_index}",
								"headOffset": f"{span.start_index}-{span.end_index}"}
span_elem = ET.Element("Term", attrib=span_attribs, text=span.text)

event = anns.events[0]
predicate, subject, object = event.spans
subj_elem = ET.Element("Subject", attrib={"idref": subject.id})
obj_elem = ET.Element("Object", attrib={"idref": object.id})
event_attribs = {"id": event.id,
							   "predicate": predicate.id,
								 "type": predicate.type,
								 "indicatorType": predicate.attributes["indicatorType"]}
event = ET.Element("Event", attrib=event_attribs)
event.insert(0, subj_elem)
event.insert(1, obj_elem)
```


4. Run SemRep Factuality

```
cd Bio-SCoRes
bash bin/factualityPipeline xml_output_dir factuality_outdir 2> factuality.log
```

`factuality_outdir` will contain two subdirectories, `standoff/` and `readable/` containing the factuality annotations in brat and pipe delimited formats, respectively.

You can check the progress with

```
grep "Processing" factuality.log | wc -l
```

Then, the `.ann` files in `factuality_outdir/standoff/` and the text files in `filtered_outdir` can be read using `pybrat` as described in step 1.


5. Convert factuality annotations back to separate certainty and polarity annotations, per table 5 of
"Assigning factuality values to semantic relations extracted from biomedical research literature" (Kilicoglu et al., 2017)

```
python scripts/convert_factuality_anns.py factuality_outdir/standoff factuality_outdir/converted/
```

It is helpful to have the `.ann` and `.json` files in the same directory, so we'll symlink them.

```
find filtered_outdir/ -name '*.json' -exec ln -s {} factuality_outdir/converted/ \;
```

6. Finally, the data can be summarized with 

```
python scripts/summarize_semrep_data.py factuality_outdir/converted/
```

## Appendix: Packaging and using large datasets

If there are more than 100k predications, data loading can be quite slow, and a potential memory hog. To get around this,
we'll use [webdataset](), which is able to read samples on the fly. 

```
python datatools.py tar path/to/config.yaml outdir/
```

This will create three `.tar` files: `{train,val,test}.tar`, which contain 80%, 10%, and 10% of the total examples, respecitvely.

Separate dataset and datamodule classes are used for using tar-packaged datasets.

```python
from config import config
import data.datasets as DS
import data.datamodules as DM

config.load_yaml(path_to_yaml_file)

ds = DS.SemRepFactWebDataset.from_config(config)
dm = DM.SemRepFactWebDataModule.from_config(config)
```

The `run.py` script will automatically determine whether you're using a tar dataset or not
by checking the contents of `config.Data.datadir`.
