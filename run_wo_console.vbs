' Run TickLab-AlgoBackTune without a console window (Windows).

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

On Error Resume Next
sh.Run "pyw -3 -m ticklab_algo", 0, False
If Err.Number <> 0 Then
    Err.Clear
    sh.Run "pythonw -m ticklab_algo", 0, False
End If
If Err.Number <> 0 Then
    MsgBox "Python not found. Install Python 3.10+ and run:" & vbCrLf & _
           "pip install numpy numba", vbExclamation, "TickLab-AlgoBackTune"
End If
