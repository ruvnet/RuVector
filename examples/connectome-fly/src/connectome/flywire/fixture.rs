//! Hand-authored 100-neuron fixture in FlyWire v783 TSV format.
//!
//! The fixture lives as three `&'static str` constants so the ingest
//! tests can materialize temp TSV files without any network download
//! or large on-disk asset. The composition targets:
//!
//! - **Cell-type coverage**: KC, MBON, PN, DN, Motor, PR, LN, optic
//!   intrinsic — the classes the outer `NeuronClass` enum can map to.
//! - **NT coverage**: ACH, GLUT, GABA, HIST, SER, DOP, OCT — every
//!   entry in the research-doc §4 NT table at least once.
//! - **Side / flow coverage**: left + right, afferent + efferent +
//!   intrinsic.
//! - **Synapse shape**: 159 directed edges, file-declared ordering, no
//!   dangling references and no authored self-loops.
//!
//! `EXPECTED_*` constants capture the counts so tests can assert
//! structural invariants without re-counting rows by hand.

/// Number of neuron rows emitted by [`neurons_tsv`].
pub const EXPECTED_NEURONS: usize = 100;

/// Number of synapse rows emitted by [`connections_tsv`].
pub const EXPECTED_SYNAPSES: usize = 159;

/// Number of classification rows emitted by [`classification_tsv`]. A
/// strict subset of neurons — the loader must still function when a
/// neuron has no classification override.
pub const EXPECTED_CLASSIFICATIONS: usize = 40;

// ---------------------------------------------------------------------
// Fixture payloads.
//
// Split into const `&str` slices and `concat!`-assembled so each const
// stays under ~100 lines of source. Data is hand-authored; the 8-digit
// neuron ids are arbitrary but unique.
// ---------------------------------------------------------------------

const NEURONS_HEADER: &str =
    "neuron_id\tsupervoxel_id\tcell_type\tnt_type\tside\tnerve\tflow\tsuper_class\n";

const NEURONS_A: &str = "\
10000001\t9000001\tPR_R1\tHIST\tleft\tOCN\tafferent\tsensory\n\
10000002\t9000002\tPR_R1\tHIST\tright\tOCN\tafferent\tsensory\n\
10000003\t9000003\tPR_R7\tHIST\tleft\tOCN\tafferent\tsensory\n\
10000004\t9000004\tPR_R8\tHIST\tright\tOCN\tafferent\tsensory\n\
10000005\t9000005\tPN_glom_DA1\tACH\tleft\tAN\tafferent\tsensory\n\
10000006\t9000006\tPN_glom_DL3\tACH\tright\tAN\tafferent\tsensory\n\
10000007\t9000007\tPN_glom_VM7\tACH\tleft\tAN\tafferent\tsensory\n\
10000008\t9000008\tORN_chm_A\tACH\tleft\tAN\tafferent\tsensory\n\
10000009\t9000009\tORN_chm_B\tACH\tright\tAN\tafferent\tsensory\n\
10000010\t9000010\tJO_mech_a\tACH\tleft\tJN\tafferent\tsensory\n\
10000011\t9000011\tJO_mech_b\tACH\tright\tJN\tafferent\tsensory\n\
10000012\t9000012\tML_mech_c\tACH\tleft\tLN\tafferent\tsensory\n\
10000013\t9000013\tKC_g\tACH\tleft\t\tintrinsic\tcentral\n\
10000014\t9000014\tKC_g\tACH\tright\t\tintrinsic\tcentral\n\
10000015\t9000015\tKC_ab\tACH\tleft\t\tintrinsic\tcentral\n\
10000016\t9000016\tKC_ab\tACH\tright\t\tintrinsic\tcentral\n\
10000017\t9000017\tKC_apbp\tACH\tleft\t\tintrinsic\tcentral\n\
10000018\t9000018\tKC_apbp\tACH\tright\t\tintrinsic\tcentral\n\
10000019\t9000019\tKC_g\tACH\tleft\t\tintrinsic\tcentral\n\
10000020\t9000020\tKC_ab\tACH\tright\t\tintrinsic\tcentral\n\
";

const NEURONS_B: &str = "\
10000021\t9000021\tKC_apbp\tACH\tleft\t\tintrinsic\tcentral\n\
10000022\t9000022\tKC_g\tACH\tright\t\tintrinsic\tcentral\n\
10000023\t9000023\tKC_ab\tACH\tleft\t\tintrinsic\tcentral\n\
10000024\t9000024\tKC_apbp\tACH\tright\t\tintrinsic\tcentral\n\
10000025\t9000025\tKC_g\tACH\tleft\t\tintrinsic\tcentral\n\
10000026\t9000026\tMBON01\tGLUT\tleft\t\tintrinsic\tcentral\n\
10000027\t9000027\tMBON02\tGLUT\tright\t\tintrinsic\tcentral\n\
10000028\t9000028\tMBON03\tGABA\tleft\t\tintrinsic\tcentral\n\
10000029\t9000029\tMBON04\tGABA\tright\t\tintrinsic\tcentral\n\
10000030\t9000030\tMBON05\tACH\tleft\t\tintrinsic\tcentral\n\
10000031\t9000031\tMBON06\tACH\tright\t\tintrinsic\tcentral\n\
10000032\t9000032\tDAN_PPL1\tDOP\tleft\t\tintrinsic\tcentral\n\
10000033\t9000033\tDAN_PPL1\tDOP\tright\t\tintrinsic\tcentral\n\
10000034\t9000034\tDAN_PAM\tDOP\tleft\t\tintrinsic\tcentral\n\
10000035\t9000035\tDAN_PAM\tDOP\tright\t\tintrinsic\tcentral\n\
10000036\t9000036\tOAN_VPM3\tOCT\tleft\t\tintrinsic\tcentral\n\
10000037\t9000037\tOAN_VPM3\tOCT\tright\t\tintrinsic\tcentral\n\
10000038\t9000038\tSER_DRN\tSER\tcenter\t\tintrinsic\tcentral\n\
10000039\t9000039\tSER_DRN\tSER\tcenter\t\tintrinsic\tcentral\n\
10000040\t9000040\tEPG_ring\tACH\tleft\t\tintrinsic\tcentral\n\
";

const NEURONS_C: &str = "\
10000041\t9000041\tEPG_ring\tACH\tright\t\tintrinsic\tcentral\n\
10000042\t9000042\tEPG_ring\tACH\tleft\t\tintrinsic\tcentral\n\
10000043\t9000043\tPEN_fan\tACH\tright\t\tintrinsic\tcentral\n\
10000044\t9000044\tPEN_fan\tACH\tleft\t\tintrinsic\tcentral\n\
10000045\t9000045\tFB_col\tACH\tright\t\tintrinsic\tcentral\n\
10000046\t9000046\tFB_col\tACH\tleft\t\tintrinsic\tcentral\n\
10000047\t9000047\tLAL_loc\tACH\tright\t\tintrinsic\tcentral\n\
10000048\t9000048\tLAL_loc\tGABA\tleft\t\tintrinsic\tcentral\n\
10000049\t9000049\tDNp01\tACH\tleft\tCN\tefferent\tdescending\n\
10000050\t9000050\tDNp02\tACH\tright\tCN\tefferent\tdescending\n\
10000051\t9000051\tDNp03\tACH\tleft\tCN\tefferent\tdescending\n\
10000052\t9000052\tDNg01\tACH\tright\tCN\tefferent\tdescending\n\
10000053\t9000053\tDNg02\tACH\tleft\tCN\tefferent\tdescending\n\
10000054\t9000054\tMotor_leg_1\tACH\tleft\tLN\tefferent\tmotor\n\
10000055\t9000055\tMotor_leg_2\tACH\tright\tLN\tefferent\tmotor\n\
10000056\t9000056\tMotor_leg_3\tACH\tleft\tLN\tefferent\tmotor\n\
10000057\t9000057\tMotor_wing_1\tACH\tright\tWN\tefferent\tmotor\n\
10000058\t9000058\tMotor_wing_2\tACH\tleft\tWN\tefferent\tmotor\n\
10000059\t9000059\tMotor_wing_3\tACH\tright\tWN\tefferent\tmotor\n\
10000060\t9000060\tMotor_hlt\tACH\tleft\tHN\tefferent\tmotor\n\
";

const NEURONS_D: &str = "\
10000061\t9000061\tLN_GABA_A\tGABA\tleft\t\tintrinsic\tcentral\n\
10000062\t9000062\tLN_GABA_B\tGABA\tright\t\tintrinsic\tcentral\n\
10000063\t9000063\tLN_GABA_C\tGABA\tleft\t\tintrinsic\tcentral\n\
10000064\t9000064\tLN_GABA_D\tGABA\tright\t\tintrinsic\tcentral\n\
10000065\t9000065\tLN_GABA_E\tGABA\tleft\t\tintrinsic\tcentral\n\
10000066\t9000066\tLN_GABA_F\tGABA\tright\t\tintrinsic\tcentral\n\
10000067\t9000067\tLN_mix_G\tGLUT\tleft\t\tintrinsic\tcentral\n\
10000068\t9000068\tLN_mix_H\tGLUT\tright\t\tintrinsic\tcentral\n\
10000069\t9000069\tLN_mix_I\tGLUT\tleft\t\tintrinsic\tcentral\n\
10000070\t9000070\tLN_mix_J\tGLUT\tright\t\tintrinsic\tcentral\n\
10000071\t9000071\tLoc_opt_A\tACH\tleft\t\tintrinsic\toptic\n\
10000072\t9000072\tLoc_opt_B\tACH\tright\t\tintrinsic\toptic\n\
10000073\t9000073\tLoc_opt_C\tACH\tleft\t\tintrinsic\toptic\n\
10000074\t9000074\tLoc_opt_D\tGABA\tright\t\tintrinsic\toptic\n\
10000075\t9000075\tLoc_opt_E\tGABA\tleft\t\tintrinsic\toptic\n\
10000076\t9000076\tLoc_opt_F\tACH\tright\t\tintrinsic\toptic\n\
10000077\t9000077\tLoc_opt_G\tGLUT\tleft\t\tintrinsic\toptic\n\
10000078\t9000078\tLoc_opt_H\tGLUT\tright\t\tintrinsic\toptic\n\
10000079\t9000079\tLoc_opt_I\tACH\tleft\t\tintrinsic\toptic\n\
10000080\t9000080\tLoc_opt_J\tGABA\tright\t\tintrinsic\toptic\n\
";

const NEURONS_E: &str = "\
10000081\t9000081\tPN_glom_DM1\tACH\tleft\tAN\tafferent\tsensory\n\
10000082\t9000082\tPN_glom_DM2\tACH\tright\tAN\tafferent\tsensory\n\
10000083\t9000083\tPN_glom_DM3\tACH\tleft\tAN\tafferent\tsensory\n\
10000084\t9000084\tAscending_A\tACH\tright\t\tintrinsic\tascending\n\
10000085\t9000085\tAscending_B\tACH\tleft\t\tintrinsic\tascending\n\
10000086\t9000086\tAscending_C\tACH\tright\t\tintrinsic\tascending\n\
10000087\t9000087\tAscending_D\tACH\tleft\t\tintrinsic\tascending\n\
10000088\t9000088\tProj_lcb_A\tACH\tleft\t\tintrinsic\tcentral\n\
10000089\t9000089\tProj_lcb_B\tACH\tright\t\tintrinsic\tcentral\n\
10000090\t9000090\tProj_lcb_C\tACH\tleft\t\tintrinsic\tcentral\n\
10000091\t9000091\tProj_lcb_D\tACH\tright\t\tintrinsic\tcentral\n\
10000092\t9000092\tProj_lcb_E\tACH\tleft\t\tintrinsic\tcentral\n\
10000093\t9000093\tMisc_X_A\tACH\tleft\t\tintrinsic\tother\n\
10000094\t9000094\tMisc_X_B\tACH\tright\t\tintrinsic\tother\n\
10000095\t9000095\tMisc_X_C\tACH\tleft\t\tintrinsic\tother\n\
10000096\t9000096\tMisc_X_D\tACH\tright\t\tintrinsic\tother\n\
10000097\t9000097\tMisc_X_E\tACH\tleft\t\tintrinsic\tother\n\
10000098\t9000098\tMisc_X_F\tACH\tright\t\tintrinsic\tother\n\
10000099\t9000099\tMisc_X_G\tACH\tleft\t\tintrinsic\tother\n\
10000100\t9000100\tMisc_X_H\tACH\tright\t\tintrinsic\tother\n\
";

/// Return the full neurons TSV payload (header + 100 data rows).
pub fn neurons_tsv() -> String {
    let mut s = String::with_capacity(12 * 1024);
    s.push_str(NEURONS_HEADER);
    s.push_str(NEURONS_A);
    s.push_str(NEURONS_B);
    s.push_str(NEURONS_C);
    s.push_str(NEURONS_D);
    s.push_str(NEURONS_E);
    s
}

const CONNECTIONS_HEADER: &str = "pre_id\tpost_id\tneuropil\tsyn_count\tsyn_weight\tnt_type\n";

const CONNECTIONS_A: &str = "\
10000001\t10000071\tME_L\t12\t12.0\tHIST\n\
10000001\t10000072\tME_L\t8\t8.0\tHIST\n\
10000002\t10000071\tME_R\t10\t10.0\tHIST\n\
10000002\t10000073\tME_R\t7\t7.0\tHIST\n\
10000003\t10000074\tME_L\t9\t9.0\tHIST\n\
10000003\t10000075\tME_L\t11\t11.0\tHIST\n\
10000004\t10000076\tME_R\t5\t5.0\tHIST\n\
10000004\t10000077\tME_R\t6\t6.0\tHIST\n\
10000005\t10000013\tMB_CA_L\t14\t14.0\tACH\n\
10000005\t10000015\tMB_CA_L\t9\t9.0\tACH\n\
10000005\t10000017\tMB_CA_L\t7\t7.0\tACH\n\
10000006\t10000014\tMB_CA_R\t13\t13.0\tACH\n\
10000006\t10000016\tMB_CA_R\t11\t11.0\tACH\n\
10000006\t10000018\tMB_CA_R\t8\t8.0\tACH\n\
10000007\t10000013\tMB_CA_L\t6\t6.0\tACH\n\
10000007\t10000019\tMB_CA_L\t5\t5.0\tACH\n\
10000008\t10000013\tMB_CA_L\t10\t10.0\tACH\n\
10000008\t10000020\tMB_CA_R\t4\t4.0\tACH\n\
10000009\t10000014\tMB_CA_R\t12\t12.0\tACH\n\
10000009\t10000021\tMB_CA_L\t3\t3.0\tACH\n\
10000010\t10000022\tMB_CA_R\t8\t8.0\tACH\n\
10000010\t10000025\tMB_CA_L\t4\t4.0\tACH\n\
10000011\t10000023\tMB_CA_L\t7\t7.0\tACH\n\
10000011\t10000024\tMB_CA_R\t6\t6.0\tACH\n\
10000012\t10000025\tMB_CA_L\t5\t5.0\tACH\n\
10000081\t10000013\tMB_CA_L\t9\t9.0\tACH\n\
10000081\t10000015\tMB_CA_L\t6\t6.0\tACH\n\
10000082\t10000014\tMB_CA_R\t11\t11.0\tACH\n\
10000082\t10000016\tMB_CA_R\t8\t8.0\tACH\n\
10000083\t10000017\tMB_CA_L\t5\t5.0\tACH\n\
10000083\t10000019\tMB_CA_L\t7\t7.0\tACH\n\
";

const CONNECTIONS_B: &str = "\
10000013\t10000026\tMB_LH_L\t4\t4.0\tACH\n\
10000013\t10000030\tMB_LH_L\t3\t3.0\tACH\n\
10000014\t10000027\tMB_LH_R\t5\t5.0\tACH\n\
10000014\t10000031\tMB_LH_R\t4\t4.0\tACH\n\
10000015\t10000026\tMB_LH_L\t6\t6.0\tACH\n\
10000015\t10000028\tMB_LH_L\t3\t3.0\tACH\n\
10000016\t10000027\tMB_LH_R\t5\t5.0\tACH\n\
10000016\t10000029\tMB_LH_R\t4\t4.0\tACH\n\
10000017\t10000030\tMB_LH_L\t3\t3.0\tACH\n\
10000018\t10000031\tMB_LH_R\t5\t5.0\tACH\n\
10000019\t10000028\tMB_LH_L\t6\t6.0\tACH\n\
10000020\t10000029\tMB_LH_R\t4\t4.0\tACH\n\
10000021\t10000030\tMB_LH_L\t5\t5.0\tACH\n\
10000022\t10000031\tMB_LH_R\t7\t7.0\tACH\n\
10000023\t10000026\tMB_LH_L\t3\t3.0\tACH\n\
10000024\t10000027\tMB_LH_R\t4\t4.0\tACH\n\
10000025\t10000030\tMB_LH_L\t6\t6.0\tACH\n\
10000032\t10000013\tMB_PPL1_L\t3\t3.0\tDOP\n\
10000033\t10000014\tMB_PPL1_R\t4\t4.0\tDOP\n\
10000034\t10000015\tMB_PAM_L\t3\t3.0\tDOP\n\
10000035\t10000016\tMB_PAM_R\t4\t4.0\tDOP\n\
10000036\t10000017\tMB_OA_L\t2\t2.0\tOCT\n\
10000037\t10000018\tMB_OA_R\t3\t3.0\tOCT\n\
10000038\t10000040\tEB_L\t2\t2.0\tSER\n\
10000039\t10000041\tEB_R\t2\t2.0\tSER\n\
10000040\t10000044\tEB_L\t5\t5.0\tACH\n\
10000041\t10000043\tEB_R\t4\t4.0\tACH\n\
10000042\t10000044\tEB_L\t6\t6.0\tACH\n\
10000043\t10000045\tFB_L\t4\t4.0\tACH\n\
10000044\t10000046\tFB_L\t5\t5.0\tACH\n\
10000045\t10000047\tLAL_L\t6\t6.0\tACH\n\
10000046\t10000048\tLAL_R\t4\t4.0\tACH\n\
";

const CONNECTIONS_C: &str = "\
10000047\t10000049\tLAL_L\t5\t5.0\tACH\n\
10000048\t10000050\tLAL_R\t4\t4.0\tGABA\n\
10000026\t10000049\tSMP_L\t6\t6.0\tGLUT\n\
10000027\t10000050\tSMP_R\t5\t5.0\tGLUT\n\
10000028\t10000049\tSMP_L\t3\t3.0\tGABA\n\
10000029\t10000050\tSMP_R\t4\t4.0\tGABA\n\
10000030\t10000051\tSMP_L\t5\t5.0\tACH\n\
10000031\t10000052\tSMP_R\t4\t4.0\tACH\n\
10000049\t10000054\tGNG_L\t8\t8.0\tACH\n\
10000049\t10000056\tGNG_L\t5\t5.0\tACH\n\
10000050\t10000055\tGNG_R\t7\t7.0\tACH\n\
10000050\t10000057\tGNG_R\t4\t4.0\tACH\n\
10000051\t10000058\tGNG_L\t5\t5.0\tACH\n\
10000052\t10000059\tGNG_R\t4\t4.0\tACH\n\
10000053\t10000060\tGNG_L\t6\t6.0\tACH\n\
10000051\t10000054\tGNG_L\t3\t3.0\tACH\n\
10000052\t10000055\tGNG_R\t3\t3.0\tACH\n\
10000053\t10000057\tGNG_R\t4\t4.0\tACH\n\
10000061\t10000013\tMB_CA_L\t2\t2.0\tGABA\n\
10000062\t10000014\tMB_CA_R\t3\t3.0\tGABA\n\
10000063\t10000015\tMB_CA_L\t2\t2.0\tGABA\n\
10000064\t10000016\tMB_CA_R\t3\t3.0\tGABA\n\
10000065\t10000017\tMB_CA_L\t2\t2.0\tGABA\n\
10000066\t10000018\tMB_CA_R\t3\t3.0\tGABA\n\
10000067\t10000019\tAL_L\t4\t4.0\tGLUT\n\
10000068\t10000020\tAL_R\t5\t5.0\tGLUT\n\
10000069\t10000021\tAL_L\t3\t3.0\tGLUT\n\
10000070\t10000022\tAL_R\t4\t4.0\tGLUT\n\
10000005\t10000061\tAL_L\t3\t3.0\tACH\n\
10000006\t10000062\tAL_R\t3\t3.0\tACH\n\
10000007\t10000063\tAL_L\t2\t2.0\tACH\n\
10000008\t10000064\tAL_R\t2\t2.0\tACH\n\
";

const CONNECTIONS_D: &str = "\
10000009\t10000065\tAL_L\t3\t3.0\tACH\n\
10000010\t10000066\tAL_R\t3\t3.0\tACH\n\
10000081\t10000067\tAL_L\t2\t2.0\tACH\n\
10000082\t10000068\tAL_R\t2\t2.0\tACH\n\
10000083\t10000069\tAL_L\t3\t3.0\tACH\n\
10000071\t10000013\tLO_L\t4\t4.0\tACH\n\
10000072\t10000014\tLO_R\t4\t4.0\tACH\n\
10000073\t10000015\tLO_L\t3\t3.0\tACH\n\
10000074\t10000016\tLO_R\t3\t3.0\tGABA\n\
10000075\t10000017\tLO_L\t2\t2.0\tGABA\n\
10000076\t10000018\tLO_R\t3\t3.0\tACH\n\
10000077\t10000019\tLO_L\t2\t2.0\tGLUT\n\
10000078\t10000020\tLO_R\t2\t2.0\tGLUT\n\
10000079\t10000040\tLO_L\t3\t3.0\tACH\n\
10000080\t10000041\tLO_R\t3\t3.0\tGABA\n\
10000054\t10000084\tVNC_L\t6\t6.0\tACH\n\
10000055\t10000085\tVNC_R\t5\t5.0\tACH\n\
10000056\t10000086\tVNC_L\t4\t4.0\tACH\n\
10000057\t10000087\tVNC_R\t5\t5.0\tACH\n\
10000084\t10000049\tSMP_L\t3\t3.0\tACH\n\
10000085\t10000050\tSMP_R\t3\t3.0\tACH\n\
10000086\t10000051\tSMP_L\t2\t2.0\tACH\n\
10000087\t10000052\tSMP_R\t2\t2.0\tACH\n\
10000088\t10000026\tSMP_L\t4\t4.0\tACH\n\
10000088\t10000049\tSMP_L\t3\t3.0\tACH\n\
10000089\t10000027\tSMP_R\t4\t4.0\tACH\n\
10000089\t10000050\tSMP_R\t3\t3.0\tACH\n\
10000090\t10000028\tSMP_L\t3\t3.0\tACH\n\
10000090\t10000040\tSMP_L\t2\t2.0\tACH\n\
10000091\t10000029\tSMP_R\t3\t3.0\tACH\n\
10000091\t10000041\tSMP_R\t2\t2.0\tACH\n\
10000092\t10000030\tSMP_L\t3\t3.0\tACH\n\
";

const CONNECTIONS_E: &str = "\
10000092\t10000043\tSMP_L\t2\t2.0\tACH\n\
10000093\t10000013\tGNG_L\t1\t1.0\tACH\n\
10000094\t10000014\tGNG_R\t1\t1.0\tACH\n\
10000095\t10000015\tGNG_L\t1\t1.0\tACH\n\
10000096\t10000016\tGNG_R\t1\t1.0\tACH\n\
10000097\t10000017\tGNG_L\t1\t1.0\tACH\n\
10000098\t10000018\tGNG_R\t1\t1.0\tACH\n\
10000099\t10000019\tGNG_L\t1\t1.0\tACH\n\
10000100\t10000020\tGNG_R\t1\t1.0\tACH\n\
10000032\t10000026\tMB_MBON_L\t2\t2.0\tDOP\n\
10000033\t10000027\tMB_MBON_R\t2\t2.0\tDOP\n\
10000034\t10000028\tMB_MBON_L\t2\t2.0\tDOP\n\
10000035\t10000029\tMB_MBON_R\t2\t2.0\tDOP\n\
10000036\t10000030\tMB_MBON_L\t1\t1.0\tOCT\n\
10000037\t10000031\tMB_MBON_R\t1\t1.0\tOCT\n\
10000058\t10000084\tVNC_L\t3\t3.0\tACH\n\
10000059\t10000085\tVNC_R\t3\t3.0\tACH\n\
10000060\t10000086\tVNC_L\t2\t2.0\tACH\n\
10000026\t10000040\tSMP_L\t3\t3.0\tGLUT\n\
10000027\t10000041\tSMP_R\t3\t3.0\tGLUT\n\
10000028\t10000040\tSMP_L\t2\t2.0\tGABA\n\
10000029\t10000041\tSMP_R\t2\t2.0\tGABA\n\
10000030\t10000042\tSMP_L\t3\t3.0\tACH\n\
10000031\t10000043\tSMP_R\t3\t3.0\tACH\n\
10000067\t10000026\tAL_L\t2\t2.0\tGLUT\n\
10000068\t10000027\tAL_R\t2\t2.0\tGLUT\n\
10000069\t10000028\tAL_L\t2\t2.0\tGLUT\n\
10000070\t10000029\tAL_R\t2\t2.0\tGLUT\n\
10000071\t10000026\tLO_L\t2\t2.0\tACH\n\
10000072\t10000027\tLO_R\t2\t2.0\tACH\n\
10000073\t10000028\tLO_L\t2\t2.0\tACH\n\
10000074\t10000029\tLO_R\t2\t2.0\tGABA\n\
";

/// FlyWire-format connections TSV (header + 260 data rows).
pub fn connections_tsv() -> String {
    let mut s = String::with_capacity(16 * 1024);
    s.push_str(CONNECTIONS_HEADER);
    s.push_str(CONNECTIONS_A);
    s.push_str(CONNECTIONS_B);
    s.push_str(CONNECTIONS_C);
    s.push_str(CONNECTIONS_D);
    s.push_str(CONNECTIONS_E);
    s
}

const CLASSIFICATION_HEADER: &str = "neuron_id\tcell_type\tsuper_class\n";

/// FlyWire-format classification TSV (40 authoritative overrides).
const CLASSIFICATION_BODY: &str = "\
10000013\tKC_g\tcentral\n\
10000014\tKC_g\tcentral\n\
10000015\tKC_ab\tcentral\n\
10000016\tKC_ab\tcentral\n\
10000017\tKC_apbp\tcentral\n\
10000018\tKC_apbp\tcentral\n\
10000019\tKC_g\tcentral\n\
10000020\tKC_ab\tcentral\n\
10000021\tKC_apbp\tcentral\n\
10000022\tKC_g\tcentral\n\
10000026\tMBON01\tcentral\n\
10000027\tMBON02\tcentral\n\
10000028\tMBON03\tcentral\n\
10000029\tMBON04\tcentral\n\
10000030\tMBON05\tcentral\n\
10000031\tMBON06\tcentral\n\
10000049\tDNp01\tdescending\n\
10000050\tDNp02\tdescending\n\
10000051\tDNp03\tdescending\n\
10000052\tDNg01\tdescending\n\
10000053\tDNg02\tdescending\n\
10000054\tMotor_leg_1\tmotor\n\
10000055\tMotor_leg_2\tmotor\n\
10000056\tMotor_leg_3\tmotor\n\
10000057\tMotor_wing_1\tmotor\n\
10000058\tMotor_wing_2\tmotor\n\
10000059\tMotor_wing_3\tmotor\n\
10000060\tMotor_hlt\tmotor\n\
10000001\tPR_R1\tsensory\n\
10000002\tPR_R1\tsensory\n\
10000003\tPR_R7\tsensory\n\
10000004\tPR_R8\tsensory\n\
10000032\tDAN_PPL1\tcentral\n\
10000033\tDAN_PPL1\tcentral\n\
10000034\tDAN_PAM\tcentral\n\
10000035\tDAN_PAM\tcentral\n\
10000036\tOAN_VPM3\tcentral\n\
10000037\tOAN_VPM3\tcentral\n\
10000038\tSER_DRN\tcentral\n\
10000039\tSER_DRN\tcentral\n\
";

/// FlyWire-format classification TSV (header + 40 override rows).
pub fn classification_tsv() -> String {
    let mut s = String::with_capacity(2 * 1024);
    s.push_str(CLASSIFICATION_HEADER);
    s.push_str(CLASSIFICATION_BODY);
    s
}

/// Write the three fixture TSVs to `dir`, returning the paths of
/// `(neurons, connections, classification)`. The files are named
/// `neurons.tsv`, `connections.tsv`, `classification.tsv` — the same
/// names used on the FlyWire release.
pub fn write_fixture(dir: &std::path::Path) -> std::io::Result<FixturePaths> {
    let neurons = dir.join("neurons.tsv");
    let connections = dir.join("connections.tsv");
    let classification = dir.join("classification.tsv");
    std::fs::write(&neurons, neurons_tsv())?;
    std::fs::write(&connections, connections_tsv())?;
    std::fs::write(&classification, classification_tsv())?;
    Ok(FixturePaths {
        neurons,
        connections,
        classification,
    })
}

/// Paths to a materialized fixture, as returned by [`write_fixture`].
#[derive(Clone, Debug)]
pub struct FixturePaths {
    /// `neurons.tsv` path.
    pub neurons: std::path::PathBuf,
    /// `connections.tsv` path.
    pub connections: std::path::PathBuf,
    /// `classification.tsv` path.
    pub classification: std::path::PathBuf,
}
