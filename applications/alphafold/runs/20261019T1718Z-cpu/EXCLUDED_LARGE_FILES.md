# Files not committed (above 100 MB, or re-derivable intermediates)

Listed by `scripts/large_files.py` at 2026-10-23T09:08Z. Each is gitignored, stays on local disk under this run directory, and is re-derivable with the scripts in `scripts/` (see `data/MANIFEST.md`).

| file | bytes | sha256 |
|---|---|---|
| `data/dat_pfamgo_accessions_2026_01.txt` | 1,678,722,670 | `5a411b17a79cadbe6e12f3e1eb649071324c0dccf93c5502056d0871ea5d3b42` |
| `data/extract_2026_01/plddt_metadata_pfamgo.csv` | 1,470,718,515 | `60b82352974468b4aef1240464f7924d504838df6e30e11de413fc4b7d589477` |
| `data/extract_2026_01/shards/s0/plddt.csv` | 367,674,498 | `0b86d8a620c2f20caedab0954bae819c5f5b15335953aa6f975be5ca6c97f765` |
| `data/extract_2026_01/shards/s0/shard.dat.gz` | 1,631,726,041 | `f938741c68e500715f645418332dd6babcecbdeb1b0fcec31369981dbc42dc44` |
| `data/extract_2026_01/shards/s0/transactions.parquet` | 480,837,411 | `98151d312165287d1fc2dcba75be95e8659cd1e342f331fa1cab8287c23bc713` |
| `data/extract_2026_01/shards/s1/plddt.csv` | 367,688,510 | `f3569398425053d8dee4d036bf9eab3f33d174a51e072ab6eb70ce0ed99d1a3b` |
| `data/extract_2026_01/shards/s1/shard.dat.gz` | 1,631,915,595 | `e45bee22fe3bdba3584c690a29423007a277a76c25ab6be6e97109e5ddbd2f96` |
| `data/extract_2026_01/shards/s1/transactions.parquet` | 482,882,594 | `8844b02a57b10415a0cf5977a0170d741e4493b44ba4abd4d5159d78823249e0` |
| `data/extract_2026_01/shards/s2/plddt.csv` | 367,667,478 | `930687616e9f75aff97bfad052f036bc1b4837bec71c05c66003484690c3579d` |
| `data/extract_2026_01/shards/s2/shard.dat.gz` | 1,631,670,708 | `285814f937e0a5f67e10f62d12a0b6e8c23a6a62cbbef74acc2a8cf035662da3` |
| `data/extract_2026_01/shards/s2/transactions.parquet` | 484,067,436 | `835338988ff90b714dffc41d3e7720cf05b6e76f2cc95169fd6e08d975a3de0e` |
| `data/extract_2026_01/shards/s3/plddt.csv` | 367,688,134 | `fe60658c655dc2961ea7ca3bc7a19e79d9151ca5b754300090102a71a282bf25` |
| `data/extract_2026_01/shards/s3/shard.dat.gz` | 1,631,980,566 | `5430a057cd7b44b80730e1ac5086c269596df9db0a0dad9fb62cbebf208d40f3` |
| `data/extract_2026_01/shards/s3/transactions.parquet` | 484,085,215 | `1a4869318d6ce8f828a72c9a0d00f01134aaeba3c73b1ae9b8e76e2301dc610f` |
| `data/extract_2026_01/shardtest/direct/transactions.parquet` | 23,311,701 | `94f38d491c735477c9bd66a1d656538cb86e327d35c7024ba4a38972229f8ec0` |
| `data/extract_2026_01/shardtest/merged/transactions_214m_base.parquet` | 23,206,335 | `b7dd7d90eb1ed7d2ea55b357902d76d50c8337c5a03007a8fbd9b94a73e42160` |
| `data/extract_2026_01/shardtest/s0/plddt.csv` | 8,112,647 | `ad89ff592d033a8e272b40f0c5e4e93922a7cf7ea1351e9afe5be4a26118aaa6` |
| `data/extract_2026_01/shardtest/s0/shard.dat.gz` | 36,406,562 | `9f89f14c9ead59cd83ff2235b0975f2c126bfe35da3c5d6846deb536bd04a3a1` |
| `data/extract_2026_01/shardtest/s0/transactions.parquet` | 10,546,864 | `030a698e3c7d69ca148a795d95a3460dd61a59d0638a34fcaa03409a24fecb3f` |
| `data/extract_2026_01/shardtest/s1/plddt.csv` | 8,123,212 | `e3f5d2fd46c8bf33ba97790ae7399f935eed89ebabb7635712f19c5140124797` |
| `data/extract_2026_01/shardtest/s1/shard.dat.gz` | 36,602,967 | `b32ff364af730540edf047f9b27a75f5dbb3a4919bdd928a7dcbd10154ec3825` |
| `data/extract_2026_01/shardtest/s1/transactions.parquet` | 10,585,834 | `a7ee81ce0ec52c4ec90693782cd581bf51a9fc9b80065bf3722e261662d1fa23` |
| `data/extract_2026_01/shardtest/s2/plddt.csv` | 8,127,084 | `f97113eb373c139984fe407736e70887051142f0145eb3ba0dbd8e0e92939a71` |
| `data/extract_2026_01/shardtest/s2/shard.dat.gz` | 36,188,790 | `d46710c725df6ed547a9473d2381c0ed5d78fa790950d3d3a711363110f932a2` |
| `data/extract_2026_01/shardtest/s2/transactions.parquet` | 10,529,893 | `468d629247ee8d53edb9edbd7855fd1c04cca73604495bf88e3975f21ddf4d11` |
| `data/extract_2026_01/shardtest/s3/plddt.csv` | 8,113,962 | `f0de1968c8f406073113e131274d7adb4bdc87bf4516534aeacc531c72e5806a` |
| `data/extract_2026_01/shardtest/s3/shard.dat.gz` | 36,922,822 | `632b1d640b845f101db8017e423e79bbe57c4721a17c331cdc2a8f988903ac8a` |
| `data/extract_2026_01/shardtest/s3/transactions.parquet` | 10,632,392 | `d8ed1103b1caf7c2fff887f668b3d8176fcda4f1c1d73e9fa7e05c7579c57296` |
| `data/extract_2026_01/shardtest/sample.csv` | 32,476,800 | `439fdc5fb8bbd40008c1c274ffd2cb11fa6c8a1d7e42d94469b945c8002ec4b4` |
| `data/extract_2026_01/shardtest/sample.dat.gz` | 145,907,904 | `44f1ffa44cb204d4b38c820216784be8ebdc6eacf114e63f323ed8880d50c041` |
| `data/extract_2026_01/transactions_214m_base.parquet` | 1,306,842,460 | `86ee18220a2a336be30f49bb59d3b202d2ad2ad76cf90969314dbfdd1bb82cdf` |
| `data/extract_2026_01/transactions_214m_base_multi.parquet` | 888,932,921 | `81867413dc5f53efa93978c0dc549cdeb57df653ecb0ccee6bfacf498e8ffaa1` |
| `data/plddt_metadata_storageapi.csv` | 3,495,993,272 | `6068f714550a0d6c13a90d215b68486b22893b489d8203d1d6db823fe375919e` |
| `data/uniprot_trembl_2026_01.csvacc.dat.gz` | 6,447,078,382 | `90496a0ba519650a020666ee0e0ef0e708c8f780f45c625f81e24352e4339807` |
| `data/uniprot_trembl_2026_01.pfamgo.dat.gz` | 10,580,479,054 | `5d282199a1251b9e17a6f1576f0a05385bec8ffa80a9a2520787d6896335cacf` |
| `data/uniprot_trembl_2026_01.reduced.dat.gz` | 27,116,638,807 | `99a59f01d2c9e899173dc1aa0e5498c0fa27ed41298f595e05cd4a0d5ee2e7ff` |

## Files kept out of main

On main these runs live under `applications/alphafold/runs/`. Committed files that were left out there (itemset tables,
third-party data copies, state markers) are listed with size and sha256 in `applications/alphafold/EXCLUDED_FROM_MAIN.md`.
