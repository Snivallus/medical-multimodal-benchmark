# Image triage

Every shipped image described by `Qwen3.8-Max` without showing it the answer key,
then compared against the gold labels locally. Produced by `bench/triage_images.py`,
which has since been removed from the repository (as have the raw per-image records);
this file is the record that is left.

This file describes the plates **before** the hand crop: 61 of the 155 shipped
images were later cropped by hand (`9cc1a78`) and were not described again, so a
C-tier row can now point at a caption that is no longer on the file, and an item
may have dropped a tier. An automated crop (`bench/strip_captions.py`, also
removed) was tried first and reverted in favour of cropping by hand.

**The descriptions are from the first pass.** Plates that were cropped or swapped by
hand afterwards have not been re-described — the vision quota is exhausted — so a
row can describe a caption that is no longer on the file, and an image added by hand
has no description at all (it is counted as B, "needs a look"). Treat the tier counts
below as approximate: re-describing them needs a fresh key and the triage script back.

Tiers, worst image in the item decides:

- **A** 影像 / 病理切片 / 内镜, text only labels or absent — usable as visual evidence.
- **B** 临床照片, 示意/其他, or unreadable — weakly usable, needs a human look.
- **C** 检验报告, 病历文本截图, transcribed text states the conclusion, or the gold
  label is printed on the image — the photo condition is really OCR of the answer.

Before the hand crop: items **46 A / 3 B / 36 C** (85 total), images 96 A / 5 B / 54 C (155 total).

## Items to fix first (C)

| item | main diagnosis | image | modality | text kind | leak | text |
| --- | --- | --- | --- | --- | --- | --- |
| iiyi-0006 | 喉颗粒细胞瘤 | img_00.jpg | 内镜 | 结论 |  | 术后一个月 |
| iiyi-0028 | 骨纤维异常增殖症 | img_00.jpg | 影像 | 结论 | yes | 图1 男,20岁,骨纤维异常增殖症并发重度脊柱后凸畸形 1a、1b、术前10d胸腰椎正侧位X线片提示T7、T8融合椎,T10椎体楔形变,Co… |
| iiyi-0044 | 上半规管裂综合征 | img_00.jpg | 检验报告 | 标注 |  | 125 250 500 1k 2k 4k 8k -10 0 10 20 30 40 50 60 70 80 90 100 110 120 听… |
| iiyi-0038 | 背部弹力纤维瘤 | img_01.jpg | 病理切片 | 结论 |  | 图2 该患者弹力纤维瘤病理结果（HE染色，×200） |
| iiyi-0060 | 尿毒症转移性软组织钙化，累及右腮腺 | img_00.jpg | 影像 | 结论 |  | 图 1 CT 示右腮腺内可见软组织钙化影 |
| iiyi-0060 | 尿毒症转移性软组织钙化，累及右腮腺 | img_01.jpg | 病理切片 | 结论 |  | 图 2 病理为瘤样钙盐沉积伴多核细胞及组织细胞反应 (×100) |
| iiyi-0063 | 色素沉着绒毛结节性滑膜炎 | img_00.jpg | 影像 | 结论 |  | 图1 左膝MRI示：左膝关节腔及腘窝内见多发结节状长T1信号影，部分结节信号混杂，左侧胫骨平台-近段及股骨远端骨质皮质毛糙欠光整，局部骨质信… |
| iiyi-0065 | 左股骨远端高分化型软骨肉瘤 | img_00.jpg | 影像 | 结论 |  | 图1 左膝关节正侧位X线片，股骨远端可见大范围骨质破坏，考虑为骨巨细胞瘤 L L |
| iiyi-0065 | 左股骨远端高分化型软骨肉瘤 | img_01.jpg | 影像 | 结论 |  | 图3 左股骨远端CT平扫：左股骨骨量增多、左股骨远端骨质破坏伴硬化、骨膜反应。图4、5 增强扫描可见不均匀强化，血管成像未波及腘动脉血管 图… |
| iiyi-0072 | 食管肺畸形 | img_00.jpg | 影像 | 结论 |  | 图1 纵隔略右移，右主支气管和各叶、段支气管显示不清，右肺实变，实变肺组织内含气支气管走行异常 图2 连续层面观察于稍低位置似见稍粗气管影与… |
| iiyi-0093 | 膀胱平滑肌瘤 | img_00.jpg | 影像 | 结论 |  | A B C 图 1 CT 泌尿 CTU 造影 A:CT 平扫(膀胱内可见类圆形低密度肿块);B:CT 增强(肿块表现为轮廓清晰的充盈缺损);… |
| iiyi-0093 | 膀胱平滑肌瘤 | img_01.jpg | 病理切片 | 结论 | yes | 图2 膀胱平滑肌瘤病理组织学表现(HE,×200) A:瘤细胞与正常膀胱平滑肌细胞相似,梭形,胞质红染,与周围正常组织境界清楚;B:瘤细胞束… |
| iiyi-0095 | 淋巴细胞性垂体炎 | img_00.jpg | 影像 | 结论 |  | 图1 PET-CT扫描结果 a:部分肝包膜、肝裂间隙、腹腔部分大网膜、肠系膜及盆底腹膜不规则增厚伴放射性摄取异常增高;b:肝门、胰头区及腹膜… |
| iiyi-0095 | 淋巴细胞性垂体炎 | img_01.jpg | 病理切片 | 结论 |  | a b 图2 肝穿刺病理活组织检查 a:肝实质呈坏死性肉芽肿炎，伴嗜酸性脓肿形成;b:抗酸染色阳性(红色箭头所指) |
| iiyi-0097 | Rasmussen综合征 | img_00.jpg | 影像 | 结论 |  | a b c d e a.横断面T1 WI,b.横断面T2 WI,c.横断面FLAIR,d.冠状位FLAIR,e.长程脑电图;左侧大脑半球脑回… |
| iiyi-0044b | 左侧大脑中动脉远端动脉瘤 | img_00.jpg | 影像 | 结论 |  | 1A 1B 1C 1D 5 PMADM 120k 图1 术中B超定位下闭塞大脑中动脉远端动脉瘤 1A 入院CT 1B 术中DSA 1C 术中… |
| iiyi-0069 | 低磷性骨软化症 | img_00.jpg | 影像 | 结论 |  | a b c 图 1 a：术前胸部 CT 平扫，左侧第 10 肋骨骨质破坏伴软组织肿块；b：术前 MRI 平扫；c：术前 MRI 增强；左侧第… |
| iiyi-0069 | 低磷性骨软化症 | img_01.jpg | 病理切片 | 结论 |  | 图 2 患者术后大体标本：呈梭形，约 8 cm×5 cm×4 cm 图 3 术后病理：磷酸盐尿性间叶性肿瘤，混合结缔组织亚型（HE×100）… |
| iiyi-0065b | 腰椎管狭窄症（L3/4、L4/5、L5/S1椎间… | img_01.jpg | 影像 | 结论 |  | 图1 术前MRI矢状位及轴位显示L4椎Ⅰ度滑脱，L3/4、L4/5、L5/S1椎间盘突出，硬脊膜囊及神经根受压 图2 术后3个月MRI矢状位… |
| iiyi-0073b | 非特指性右乳弥漫大B细胞淋巴瘤 | img_00.jpg | 临床照片 | 结论 |  | 图 1 男性原发乳腺淋巴瘤患者治疗前乳腺：右乳乳头溢液结痂，右乳外侧可触及 5 ~ 6 cm 肿块，质硬，可移动 图 2 男性原发乳腺淋巴瘤… |
| iiyi-0074b | 嗜伊红淋巴肉芽肿（木村病） | img_00.jpg | 病理切片 | 结论 |  | 1A 1B 2A 2B 2C 图1A 上睑肿胀、泪腺增大、上、下直肌肥厚（白色箭头所指） 图1B 上睑肿胀、泪腺增大、外直肌肥厚严重（白色箭… |
| iiyi-0076 | 右股骨下端骨软骨瘤 | img_00.jpg | 影像 | 结论 | yes | (1)伤后当日 (2)伤后4年 (3)术后 B C (4)术后病理组织切片图(HE染色 ×200) 图1 股骨下端骨软骨瘤治疗前后膝关节正、… |
| iiyi-0077b | 人工全髋关节置换术后假体反复脱位 | img_00.jpg | 影像 | 结论 |  | 图1 患者，女，45岁，行全髋关节翻修术 1a.初次全髋关节置换术后骨盆正位X线片示假体未脱位，坐骨支线性骨折 1b.初次全髋关节置换术后2… |
| iiyi-0082 | 下颌下静脉畸形伴多发静脉石 | img_01.jpg | 临床照片 | 结论 |  | 图 4 术中可见结石 |
| iiyi-0082 | 下颌下静脉畸形伴多发静脉石 | img_02.jpg | 临床照片 | 结论 |  | 7 8 9 10 11 12 13 14 图 5 取出的大小不一的结石 |
| iiyi-0083b | 腮腺上皮样血管内皮瘤 | img_01.jpg | 影像 | 结论 |  | 图2 3月后病变复发 CT 影像 |
| iiyi-0083b | 腮腺上皮样血管内皮瘤 | img_02.jpg | 病理切片 | 结论 |  | A B C 图3 HE染色(A)后可见少量的异型血管源性细胞,同时免疫组化(B,C)结果可见CD31(+),IE-40(+) (×40) |
| iiyi-0079 | PHACES综合征（头面部阶段性巨大血管瘤+后颅… | img_00.jpg | 临床照片 | 结论 |  | a b c 图1 患儿女，2个月11 d，PHACES综合征 a：治疗前；b：治疗后6个月；c：治疗后1年随访 |
| iiyi-0078 | PHACES综合征（头面部阶段性巨大血管瘤+Da… | img_00.jpg | 临床照片 | 结论 |  | 2012/12/2 2012/12/20 a 2012.12.30 b c 图3 患儿女，32 d，PHACES综合征 a：治疗前；b：治疗… |
| iiyi-0078 | PHACES综合征（头面部阶段性巨大血管瘤+Da… | img_01.jpg | 影像 | 结论 |  | 2012/12/20 16:36 图4 CT示左侧海绵窦旁及左侧桥小脑角区多发血管瘤；左侧脉络丛血管迂曲增粗；Dandy-Walker综合征… |
| iiyi-0090 | 右侧乳腺小管癌 | img_00.jpg | 影像 | 结论 |  | 图1 例1。乳腺X线（MLO位点压图）显示右乳1点钟方向高密度卵圆形肿块影（箭头），部分周围见大于肿块直径的长毛刺（白箭） 图2 例2。彩超… |
| iiyi-0087 | 左侧上颌高位阻生第三磨牙与多生牙融合 | img_01.jpg | 临床照片 | 结论 |  | 图3 离体融合牙图片及剖面图 |
| iiyi-0093b | 新生儿朗格汉斯细胞组织细胞增生症 | img_01.jpg | 病理切片 | 结论 |  | 图2 皮肤组织病理及免疫组织化学染色（×200） A：皮肤组织病理（苏木精-伊红染色），表皮增厚，真皮内以组织样及泡沫样细胞为主的浸润，同时… |
| iiyi-0095b | 十二指肠梗阻 | img_00.jpg | 影像 | 结论 |  | 图 1 术前上消化道造影 图 2 术后上消化道造影 ① ② |
| iiyi-0094 | 阴茎巨大表皮样囊肿 | img_00.jpg | 临床照片 | 结论 | yes | 图 1 阴茎巨大表皮样囊肿 阴茎背侧肿物，边界清楚，表面光滑，皮肤颜色为淡兰色 |
| iiyi-0094 | 阴茎巨大表皮样囊肿 | img_01.jpg | 影像 | 结论 |  | 图 2 阴茎 MRI 检查 阴茎背部见囊性异常信号影，信号呈短 T1、短 T2，DWI 呈低信号，T1-STIR 呈高信号，T2-STIR … |
| iiyi-0094 | 阴茎巨大表皮样囊肿 | img_02.jpg | 临床照片 | 结论 |  | 图3 切除的阴茎表皮样囊肿 肿物大小约8 cm×3 cm，质柔韧，包膜完整，其内为棕褐色粘稠糊状物，未发现明确组织结构 |
| iiyi-0094 | 阴茎巨大表皮样囊肿 | img_03.jpg | 病理切片 | 结论 |  | 图 4 阴茎表皮样囊肿（HE ×40） 囊壁内衬复层鳞状上皮，腔内有大量红染角化物 |
| iiyi-0096 | 颈椎椎板骨软骨瘤 | img_00.jpg | 影像 | 结论 |  | 图1 患者，男，44岁，C6椎板软骨瘤 1a.术前颈椎横断面CT扫描示C6左侧椎板骨性赘生物突入椎管压迫硬膜囊 1b.术前颈椎矢状面CT扫描… |
| iiyi-0097b | 肺腺癌 | img_00.jpg | 影像 | 结论 |  | 图1 肺癌术前CT，可见右侧肺尖部占位 |
| iiyi-0097b | 肺腺癌 | img_01.jpg | 影像 | 结论 |  | 图2 肺癌术前 PET-CT，诊断肺癌同时排除全身转移 |
| iiyi-0097b | 肺腺癌 | img_02.jpg | 影像 | 结论 |  | 图3 化疗前胸部CT, 提示手术效果良好 图4 右侧耳下区肿胀 |
| iiyi-0097b | 肺腺癌 | img_03.jpg | 影像 | 结论 |  | 图 5 下颌骨部 CT，提示下颌骨升支部溶骨性破坏 |
| iiyi-0100 | /1畸形舌、唇面沟 | img_00.jpg | 影像 | 结论 |  | 图1 根尖X线片示/1牙根被一纵向线状透射影（↑）分为近中、远中两个“牙根”，远中小牙根根尖牙周膜增宽 |
| iiyi-0100 | /1畸形舌、唇面沟 | img_01.jpg | 影像 | 结论 |  | 图2 多层螺旋CT轴位示 1/1 牙根唇、舌面均有牙体组织凹陷沟裂（↑），沟裂从颈部向根尖方向逐渐加深，最后在根中1/3分为近中牙根和远中“… |
| iiyi-0103 | 狼疮肾炎（II型+V型） | img_00.jpg | 病理切片 | 结论 |  | C D G H 注：A：HE染色可见系膜细胞数目增多（×200）；B：PAS染色见系膜细胞增多及基质增宽（×400）；C：PASM染色可见基… |
| iiyi-0106 | 石骨症 | img_00.jpg | 影像 | 结论 |  | L ① ② 图1 恶性石骨症患儿骨盆X线平片图 图2 恶性石骨症患儿头颅CT平扫图(骨窗,矢状位) |
| iiyi-0106 | 石骨症 | img_01.jpg | 影像 | 结论 |  | 图 3 恶性石骨症患儿头颅 CT 平扫图（骨窗，冠状位） H A L |
| iiyi-0107 | 右侧手掌第2、3掌骨间水银异物肉芽肿 | img_00.jpg | 影像 | 结论 |  | 1 2 3 图1 CT平扫提示右侧手掌第2、3掌骨间软组织内水银珠存留； 图2 X线片显示右侧手掌第2、3掌骨间软组织内散在水银珠； 图3 … |
| iiyi-0113 | 急性创伤性臀小肌撕裂 | img_00.jpg | 影像 | 结论 |  | 图 1 X线片显示左髋关节及左髂骨未见骨质异常 图 2 MRI显示臀小肌在髂骨止点处连续性消失(箭头处) |
| iiyi-0111 | 下颌骨结核 | img_00.jpg | 影像 | 结论 |  | 图1 患者，男，65岁，全颌曲面断层片提示左侧下颌升支骨破坏，边界不清 |
| iiyi-0111 | 下颌骨结核 | img_01.jpg | 影像 | 结论 |  | 30821 图2 患者男，65岁，下颌骨CT提示左侧升支类圆形骨样破坏 |
| iiyi-0111 | 下颌骨结核 | img_02.jpg | 病理切片 | 结论 |  | 注：镜下见大量肿胀的组织细胞和朗罕氏巨细胞，并可见浆细胞。考虑颌骨结核 图3 患者男，65岁，手术切除病灶坏死组织的病理检查结果（HE ×1… |
| iiyi-0121 | 左侧股骨近端纤维结构不良 | img_00.jpg | 影像 | 结论 |  | 图1 患者,女,48岁,股骨近端纤维结构不良 1a.术前左股骨正侧位X线片示股骨近段形态稍欠规整,股骨髓腔内密度不均匀 1b.术前髋关节核磁… |

## Items needing a look (B)

| item | main diagnosis | image | modality | findings |
| --- | --- | --- | --- | --- |
| iiyi-0049 | 安氏Ⅲ类亚类错牙合畸形 | img_00.jpg | 临床照片 | 五张口内像显示上下颌牙列、牙龈及咬合关系：正面像可见前牙覆合覆盖及中线关系，左、右侧像可见后牙咬合接触，上、下颌面像可见… |
| iiyi-0068 | 面颈部及躯干、四肢深Ⅱ～Ⅲ度水蒸气烧伤，总面积4… | img_00.jpg | 示意/其他 | 组合图：左图 CT 轴位纵隔窗示气管腔内软组织密度影（红箭）；中图内镜下见管腔内圆形、表面光滑的隆起性病变（黑箭）；右图… |
| iiyi-0086b | 左侧髌骨骨巨细胞瘤（Ⅱ级） | img_00.jpg | 示意/其他 | 九宫格组合图：上排膝关节X线（侧位、轴位）及CT断层示髌骨膨胀性、溶骨性骨质破坏，骨皮质变薄；中排术中照片示骨内红褐色病… |

## Items that survive (A)

`iiyi-0001`, `iiyi-0002`, `iiyi-0003`, `iiyi-0005`, `iiyi-0007`, `iiyi-0009`, `iiyi-0008`, `iiyi-0012`, `iiyi-0013`, `iiyi-0014`, `iiyi-0017`, `iiyi-0020`, `iiyi-0021`, `iiyi-0024`, `iiyi-0025`, `iiyi-0032`, `iiyi-0037`, `iiyi-0036`, `iiyi-0043`, `iiyi-0045`, `iiyi-0046`, `iiyi-0052`, `iiyi-0053`, `iiyi-0059`, `iiyi-0057`, `iiyi-0070`, `iiyi-0067`, `iiyi-0074`, `iiyi-0073`, `iiyi-0077`, `iiyi-0083`, `iiyi-0086`, `iiyi-0089`, `iiyi-0070b`, `iiyi-0075b`, `iiyi-0080`, `iiyi-0084`, `iiyi-0085`, `iiyi-0089b`, `iiyi-0088`, `iiyi-0101`, `iiyi-0104`, `iiyi-0109`, `iiyi-0112`, `iiyi-0117`, `iiyi-0122`
