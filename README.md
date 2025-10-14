# Datasets Preparation
## Convert hf-datasets parquet files to jsonl file

Routine A
``` bash
pip install datasets
cd prepare_dataset
python export_finewebedu_to_jsonl.py "{$PATH_TO_HF_FINEWEB_EDU}/data/**/*.parquet" fineweb-edu.sample-200k.jsonl 200000
cd ../RWKV-v5/
python make_data.py ../prepare_dataset/fineweb-edu.sample-200k.jsonl {$SHUFFLE_TIMES} {$CTX_LEN}
```

output log:
``` log
### Convert ../prepare_dataset/fineweb-edu.sample-200k.jsonl to fineweb-edu.sample-200k.bin/idx...
### Found 200000 non-empty lines in ../prepare_dataset/fineweb-edu.sample-200k.jsonl
Shuffle: 1 out of 1
### Building binidx...
0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000 13500 14000 14500 15000 15500 16000 16500 17000 17500 18000 18500 19000 19500 20000 20500 21000 21500 22000 22500 23000 23500 24000 24500 25000 25500 26000 26500 27000 27500 28000 28500 29000 29500 30000 30500 31000 31500 32000 32500 33000 33500 34000 34500 35000 35500 36000 36500 37000 37500 38000 38500 39000 39500 40000 40500 41000 41500 42000 42500 43000 43500 44000 44500 45000 45500 46000 46500 47000 47500 48000 48500 49000 49500 50000 50500 51000 51500 52000 52500 53000 53500 54000 54500 55000 55500 56000 56500 57000 57500 58000 58500 59000 59500 60000 60500 61000 61500 62000 62500 63000 63500 64000 64500 65000 65500 66000 66500 67000 67500 68000 68500 69000 69500 70000 70500 71000 71500 72000 72500 73000 73500 74000 74500 75000 75500 76000 76500 77000 77500 78000 78500 79000 79500 80000 80500 81000 81500 82000 82500 83000 83500 84000 84500 85000 85500 86000 86500 87000 87500 88000 88500 89000 89500 90000 90500 91000 91500 92000 92500 93000 93500 94000 94500 95000 95500 96000 96500 97000 97500 98000 98500 99000 99500 100000 100500 101000 101500 102000 102500 103000 103500 104000 104500 105000 105500 106000 106500 107000 107500 108000 108500 109000 109500 110000 110500 111000 111500 112000 112500 113000 113500 114000 114500 115000 115500 116000 116500 117000 117500 118000 118500 119000 119500 120000 120500 121000 121500 122000 122500 123000 123500 124000 124500 125000 125500 126000 126500 127000 127500 128000 128500 129000 129500 130000 130500 131000 131500 132000 132500 133000 133500 134000 134500 135000 135500 136000 136500 137000 137500 138000 138500 139000 139500 140000 140500 141000 141500 142000 142500 143000 143500 144000 144500 145000 145500 146000 146500 147000 147500 148000 148500 149000 149500 150000 150500 151000 151500 152000 152500 153000 153500 154000 154500 155000 155500 156000 156500 157000 157500 158000 158500 159000 159500 160000 160500 161000 161500 162000 162500 163000 163500 164000 164500 165000 165500 166000 166500 167000 167500 168000 168500 169000 169500 170000 170500 171000 171500 172000 172500 173000 173500 174000 174500 175000 175500 176000 176500 177000 177500 178000 178500 179000 179500 180000 180500 181000 181500 182000 182500 183000 183500 184000 184500 185000 185500 186000 186500 187000 187500 188000 188500 189000 189500 190000 190500 191000 191500 192000 192500 193000 193500 194000 194500 195000 195500 196000 196500 197000 197500 198000 198500 199000 199500 done
### Verifying result...
----------------------------------------------------------------------[fineweb-edu.sample-200k idx 0 sz 510]
Zachary Taylor, Our 12th President
Born in Virginia in 1784, Zachary Taylor was taken to Kentucky as an infant where he grew up on a large plantation. In 1808 he received his first commission as an Army officer, becoming commander of the garrison at Fort Pickering (Memphis is located there today). In 1810 he married Margaret Mackall Smith. They had one son and five daughters. Being transferred from one fort to another, Taylor gained fame as an Indian fighter
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · 
 food poisoning have all been suggested as the source of his fatal illness. He had eaten a hasty snack of iced milk, cold cherries and pickled cucumbers. Also, although it was July, he was wearing a thick coat, vest, high-collared shirt and top hat which might have caused heat stroke. He was reported to have a red, flushed face and slurred speech. He is buried in Louisville, Kentucky, at what is now the Zachary Taylor National Cemetery.
----------------------------------------------------------------------[fineweb-edu.sample-200k idx 199999 sz 9341]
Mind fitness: improving operational effectiveness and building warrior resilience.Today's complex, fluid, and unpredictable operational environment both demands more from the military in terms of mission requirements and exposes troops to more stressors and potential trauma than ever before. On the one hand, situational awareness Situation awareness or situational awareness  (SA) is the mental representation and understanding of objects, events, people, system states, interactions, environmental conditions, and other situation-specific factors affecting human performance in , mental agility,
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · 
 or references that appear in reliable, third-party publications. Alone, primary sources and sources affiliated with the subject of this article are not sufficient for an accurate encyclopedia article. and Department of Government at Georgetown University. Dr. Amishi P. Jha is an Assistant Professor in the Department of Psychology at the University of Pennsylvania (body, education) University of Pennsylvania - The home of ENIAC and Machiavelli.
Address: Philadelphia, PA, USA. .
--------------------------------------------------------------------------------
### Final fineweb-edu.sample-200k.bin/idx has 211986694 tokens, 200000 items. Dtype <class 'numpy.uint16'>

### magic_prime = 51749 (for ctxlen 4096)

--my_exit_tokens 211986694 --magic_prime 51749 --ctx_len 4096
```

Routine B
``` bash
pip install datasets pyarrow
cd prepare_dataset
python make_data_streaming_noshuffle.py "/data/malulab/datasets/fineweb-edu/data/**/*.parquet" 1 4096 --out_name fineweb-edu
```

## Start training

``` bash
cd RWKV-v7/train_temp/
```

Follow the wiki rwkv: https://wiki.rwkv.com/advance/pretrain.html#adjust-training-parameters