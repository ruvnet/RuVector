// Module: LogLevel

/* original: WL */ var WL={
  
};

/* original: _f8 */ function error_info_verbose_warning(q){
  switch(q){
    case"error":return WL.LogLevel.Error;
    case"info":return WL.LogLevel.Info;
    case"verbose":return WL.LogLevel.Verbose;
    case"warning":return WL.LogLevel.Warning;
    default:return WL.LogLevel.Info
  } /* confidence: 65% */

