## DumE

**DumE** is a small language model that can be trained and run on CPU. More of an experimentation. 
Trained on my laptop for about 45 minutes and it gave me some output:

```markdown
[Creative Writing] On a formy dray, you discover a feew etering the room bracks of continuous what poses the like jow oakes. This to magnethâ€™s at cille in the ot knighles dopecang on it will end on trars. When you sharper, and how in they clonflight the togerato algeno on and forgelo revoutionized ona consured of a you the total plant the room which are you?

[Creative Writing attlnnom] The heart oc digation anciting of cases an shivilety digently one onligate pate to grew orest. The engineerine legg trate arch
```

Doesn't make any sense probably because it's trained with a make-shift dataset and little resources. However, with a much broader data set and more training time probably about 2.5 hours on a GPU, we could get much more meaningful and maybe even coherent output.

Checkout model.py for the full definition and architecture of the model.  To train the model and save the learned weights, run `modelTrain.py` though you'll have to provide a dataset to train on or you could use the default one `A_dataset_cleaned.txt`. 