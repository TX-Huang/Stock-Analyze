; ==========================================
; 全域量化終端 v200 — Inno Setup 安裝腳本
; ==========================================
; 使用方式:
;   1. 安裝 Inno Setup: https://jrsoftware.org/isdl.php
;   2. 用 Inno Setup Compiler 開啟此檔案
;   3. 按 Build → Compile 產生安裝檔
;   4. 輸出檔案在 installer/Output/ 資料夾
; ==========================================

#define MyAppName "全域量化終端"
#define MyAppVersion "3.2.0"
#define MyAppPublisher "AI Invest HQ"
#define MyAppURL "https://github.com/TX-Huang/Stock-Analyze"
#define MyAppExeName "start.bat"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\AI Invest HQ
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=AI_Invest_HQ_v{#MyAppVersion}_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
LicenseFile=..\LICENSE.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Options:"

[Files]
; Python 嵌入式環境
Source: "..\python_embed\*"; DestDir: "{app}\python_embed"; Flags: ignoreversion recursesubdirs createallsubdirs

; 應用程式原始碼
Source: "..\app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\state.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\scheduler.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; 模組
Source: "..\ui\*"; DestDir: "{app}\ui"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\data\*"; DestDir: "{app}\data"; Excludes: "*.json,__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\analysis\*"; DestDir: "{app}\analysis"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\strategies\*"; DestDir: "{app}\strategies"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\config\*"; DestDir: "{app}\config"; Excludes: "__pycache__,auth_config.yaml"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\utils\*"; DestDir: "{app}\utils"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\bot\*"; DestDir: "{app}\bot"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\scripts\*"; DestDir: "{app}\scripts"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\tests\*"; DestDir: "{app}\tests"; Excludes: "__pycache__"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Streamlit 設定
Source: "..\.streamlit\secrets.toml.example"; DestDir: "{app}\.streamlit"; Flags: ignoreversion
Source: "..\.streamlit\config.toml"; DestDir: "{app}\.streamlit"; Flags: ignoreversion onlyifdoesntexist

; 資產
Source: "..\assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; 授權
Source: "..\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
Name: "{app}\data"
Name: "{app}\.streamlit"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Edit API Keys"; Filename: "notepad.exe"; Parameters: """{app}\.streamlit\secrets.toml"""
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "notepad.exe"; Parameters: """{app}\.streamlit\secrets.toml"""; Description: "Open API key config"; Flags: postinstall nowait skipifsilent unchecked
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Quant Terminal"; Flags: postinstall nowait skipifsilent shellexec

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  SecretsPath: String;
  ExamplePath: String;
begin
  if CurStep = ssPostInstall then
  begin
    SecretsPath := ExpandConstant('{app}\.streamlit\secrets.toml');
    ExamplePath := ExpandConstant('{app}\.streamlit\secrets.toml.example');

    { Copy template to secrets.toml if not exists }
    if not FileExists(SecretsPath) then
    begin
      if FileExists(ExamplePath) then
        CopyFile(ExamplePath, SecretsPath, False);
    end;
  end;
end;
