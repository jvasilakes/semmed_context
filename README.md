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
bash bin/factualityPipeline xml_output_dir factuality_outdir
```

`factuality_outdir` will contain two subdirectories, `standoff/` and `readable/` containing the factuality annotations in brat and pipe delimited formats, respectively.

Then, the `.ann` files in `factuality_outdir/standoff/` and the text files in `filtered_outdir` can be read using `pybrat` as described in step 1.


5. Convert factuality annotations back to separate certainty and polarity annotations, per table 5 of
"Assigning factuality values to semantic relations extracted from biomedical research literature" (Kilicoglu et al., 2017)

```
python scripts/convert_factuality_anns.py factuality_outdir/standoff factuality_outdir/converted/
```

6. Finally, the data can be summarized with 

```
python scripts/summarize_semrep_data.py factuality_outdir/converted/
```
