%===============================================================================%
% Modified by: Apurv Deepak Kulkarni (25 September 2024)%
%===============================================================================%
%TODO: Check landscape mode%
\NeedsTeXFormat{LaTeX2e}%
\ProvidesClass{scadsai-poster-template}[2024/09/25 Custom Class File]%
%
\RequirePackage{kvoptions}%
\SetupKeyvalOptions{family=scadsai,prefix=scadsai@,setkeys=\kvsetkeys}%
%
\DeclareStringOption[english]{language}%
\DeclareVoidOption{german}{\kvsetkeys{scadsai}{language=german}}%
\DeclareVoidOption{english}{\kvsetkeys{scadsai}{language=english}}%
%
\newif\ifportrait%
\newif\ifanullb%
\newif\ifanull%
\newif\ifaeins%
\newif\ifazwei%
\newif\ifadrei%
\newif\ifpreview%
\newif\ifenglish%
\newif\ifgerman%
\newif\if@useSectionNum%
\newif\if@useSectionTitleCenter%
\newif\ifcomputetimeused%
%
\DeclareOption{a0b}{\anullbtrue}%
\DeclareOption{a0}{\anulltrue\anullbfalse}%
\DeclareOption{a1}{\aeinstrue\anullbfalse}%
\DeclareOption{a2}{\azweitrue\anullbfalse}%
\DeclareOption{a3}{\adreitrue\anullbfalse}%
%
\DeclareOption{landscape}{\portraitfalse}%
\DeclareOption{portrait}{\portraittrue}%
\DeclareOption{preview}{\previewtrue}%
\DeclareOption{final}{\previewfalse}%
\DeclareOption{english}{\englishtrue\germanfalse}%
\DeclareOption{german}{\germantrue\englishfalse}%
\DeclareOption{useSectionNumber}{\@useSectionNumtrue}%
\DeclareOption{useSectionTitleCenter}{\@useSectionTitleCentertrue}%
\computetimeusedtrue%
\DeclareOption{nocomputetimeused}{\computetimeusedfalse}%
%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax%
\ProcessKeyvalOptions{scadsai}

\ifx\scadsai@language\@empty\else
	\ifnum\pdfstrcmp{\scadsai@language}{german}=0
		\germantrue
		\PassOptionsToClass{ngerman}{article}
	\fi
	\ifnum\pdfstrcmp{\scadsai@language}{english}=0
		\englishtrue
		\PassOptionsToClass{english}{article}
	\fi
\fi

\LoadClass{article}%
%
%
%===============================================================================%
% Package setup (except language and font)%
%===============================================================================%
\usepackage[svgnames]{xcolor}%
\usepackage{textpos}%
\usepackage{fancyhdr}%
\usepackage{tabularx}%
\usepackage{pgffor,etoolbox}%
\usepackage{multirow}%
\usepackage{expl3}%
\usepackage{xparse}%
\usepackage[explicit]{titlesec}%
\RequirePackage{tikz}%
\usetikzlibrary{fadings,positioning,calc}%
\usepackage{amsmath}%
\usepackage{xstring}%
\ifgerman
	\RequirePackage[ngerman]{babel}%
	\usepackage[german]{isodate}%
\else
	\RequirePackage[english]{babel}%
	\usepackage[english]{isodate}%
\fi
\usepackage{enumitem}%
%
%===============================================================================%
% Language setup%
%===============================================================================%
% Language vocabulary%
\newcommand*{\englishvocabulary}{%
	\newcommand*{\scadsai@vocslide}{Slide}%
	\newcommand*{\scadsai@vocof}{of}%
	\newcommand*{\scadsai@vocpart}{Part}%
	\newcommand*{\scadsai@vocagenda}{Agenda}%
	\newcommand*{\scadsai@vocauthors}{Authors}%
	\newcommand*{\scadsai@voccontact}{Contact}%
	\newcommand*{\scadsai@vocpartners}{Partners}%
	\newcommand*{\scadsai@voccooperation}{In cooperation with}%
	\newcommand*{\scadsai@vocsponsoredby}{Funded by}%
	\newcommand*{\scadsai@vocacknowledgement}{The authors gratefully acknowledge the GWK support for funding this project by providing computing time through the Center for Information Services and HPC (ZIH) at TU Dresden on HRSK-II.}%
}%
%
\newcommand*{\germanvocabulary}{%
	\newcommand*{\scadsai@vocslide}{Folie}%
	\newcommand*{\scadsai@vocof}{von}%
	\newcommand*{\scadsai@vocpart}{Teil}%
	\newcommand*{\scadsai@vocagenda}{Gliederung}%
	\newcommand*{\scadsai@vocauthors}{Autoren}%
	\newcommand*{\scadsai@voccontact}{Kontakt}%
	\newcommand*{\scadsai@vocpartners}{Partners}%
	\newcommand*{\scadsai@vocsponsoredby}{Gefördert durch}%
	\newcommand*{\scadsai@voccooperation}{In Kooperation mit}%
	\newcommand*{\scadsai@vocacknowledgement}{Die Autoren bedanken sich bei der GWK für die Förderung dieses Projekts durch die Bereitstellung von Rechenzeit durch das Zentrum für Informationsdienste und HPC (ZIH) der TU Dresden auf HRSK-II.}%
}%
%
%===============================================================================%
% Font setup%
%===============================================================================%
\usepackage{fontspec}%
\defaultfontfeatures{Ligatures=TeX}%
\defaultfontfeatures[barlow]{%
	Path=./fonts/,%
	Extension = .ttf ,%
	UprightFont = Barlow-Regular,%
	BoldFont = Barlow-Bold,%
	ItalicFont = Barlow-Italic,%
	BoldItalicFont = Barlow-BoldItalic,%
	FontFace = {sb}{n}{Barlow-SemiBold},%
	FontFace = {md}{n}{Barlow-Medium},%
}%
\defaultfontfeatures[opensans]{%
	Path=./fonts/,%
	Extension = .ttf,%
	UprightFont = OpenSans-Regular,%
	BoldFont = OpenSans-Bold,%
	ItalicFont = OpenSans-Italic,%
	BoldItalicFont = OpenSans-BoldItalic,%
	FontFace = {sb}{n}{OpenSans-Semibold},%
	FontFace = {lt}{n}{OpenSans-Light},%
	FontFace = {lt}{it}{OpenSans-LightItalic},%
	FontFace = {r}{n}{OpenSans-Regular},%
	FontFace = {r}{it}{OpenSans-Italic},%
}%
%
% Family Switches
\newfontfamily\opensansfont{opensans}%
\newfontfamily\barlowfont{barlow}%
%
% Main font
\setmainfont{opensans}%
%
% Setting some font types %
\DeclareRobustCommand\ebseries{\fontseries{eb}\selectfont} % extra bold%
\DeclareRobustCommand\sbseries{\fontseries{sb}\selectfont} % semi bold%
\DeclareRobustCommand\ltseries{\fontseries{lt}\selectfont}  % light%
\DeclareRobustCommand\clseries{\fontseries{cl}\selectfont}%
\DeclareRobustCommand\mseries{\fontseries{md}\selectfont}%
\DeclareRobustCommand\rseries{\fontseries{r}\selectfont}%
%
% Aliasing commands to regular text commands%
\DeclareTextFontCommand{\texteb}{\ebseries} % extra bold%
\DeclareTextFontCommand{\textsb}{\sbseries} % semi bold%
\DeclareTextFontCommand{\textlt}{\ltseries} % light%
\DeclareTextFontCommand{\textcl}{\clseries} % condensed light text. Not working ??%
\DeclareTextFontCommand{\text}{\rseries} % condensed light text. Not working ??%
\DeclareTextFontCommand{\textit}{\itshape}%
%
% Redefining font size definition (note: only for A0 size)%
% TODO: Figure out how to change these font sizes as per paper size%
% 2nd value is based on original proportion from following link%
% https://tex.stackexchange.com/questions/322256/what-is-the-second-value-in-fontsize-for%
\renewcommand{\normalsize}{\fontsize{28}{33.6}\selectfont}%
%
% Smaller font sizes%
\renewcommand{\tiny}{\fontsize{14}{16.8}\selectfont} % 50% of 28pt%
\renewcommand{\scriptsize}{\fontsize{19.6}{23.52}\selectfont} % 70% of 28pt%
\renewcommand{\footnotesize}{\fontsize{22.4}{26.88}\selectfont} % 80% of 28pt%
\renewcommand{\small}{\fontsize{25.2}{30.24}\selectfont} % 90% of 28pt%
%
% Larger font sizes%
\renewcommand{\large}{\fontsize{33.6}{40.32}\selectfont} % 120% of 28pt%
\renewcommand{\Large}{\fontsize{39.2}{46.96}\selectfont} % 140% of 28pt%
\renewcommand{\LARGE}{\fontsize{47.6}{56.96}\selectfont} % 170% of 28pt%
\renewcommand{\huge}{\fontsize{56}{66.88}\selectfont} % 200% of 28pt%
\renewcommand{\Huge}{\fontsize{70}{83.68}\selectfont} % 250% of 28pt%
%
% Additional large font sizes%
\newcommand{\veryHuge}{\fontsize{84}{100.8}\selectfont} % 300% of 28pt%
\newcommand{\VeryHuge}{\fontsize{98}{117.12}\selectfont} % 350% of 28pt%
\newcommand{\VERYHuge}{\fontsize{126}{151.2}\selectfont} % 450% of 28pt%
%
% Poster font setup%
\newcommand{\setthemainfont}{\opensansfont\rseries\normalsize\selectfont}%
\newcommand{\setmaintitlefont}{\barlowfont\bfseries\fontsize{80}{96}\selectfont}%
\newcommand{\setmaintitlesubtitlefont}{\barlowfont\rseries\fontsize{54}{64.8}\selectfont}%
\newcommand{\setmaintitleauthorfont}{\opensansfont\rseries\fontsize{28}{33.6}\selectfont}%
\newcommand{\setmaintitleaffiliationfont}{\opensansfont\rseries\fontsize{20}{24}\selectfont}%
\newcommand{\setsectiontitlefont}{\barlowfont\rseries\fontsize{44}{52.8}\selectfont}%
\newcommand{\setsubsectiontitlefont}{\barlowfont\mseries\Large}%
\newcommand{\setfooterunivfont}{\opensansfont\normalsize}%
\newcommand{\setfooterauthorfont}{\opensansfont\rseries\normalsize}%
\newcommand{\setfooterauthoremailfont}{\setfooterauthorfont\normalsize}%
\newcommand{\setfootertitlefont}{\bfseries\barlowfont\fontsize{30}{30}\selectfont}%
%
%
%===============================================================================%
% Paper size and margin setup%
%===============================================================================%
% Paper size selection%
\newcommand{\SetPaperSize}[2]{%
	\setlength{\paperwidth}{#1}%
	\setlength{\paperheight}{#2}%
}%
\ifanullb%
\SetPaperSize{129.3cm}{91.4cm} % Custom large paper%
\else\ifanull%
	\SetPaperSize{118.9cm}{84.1cm} % A0%
\else\ifaeins%
		\SetPaperSize{84.1cm}{59.4cm}   % A1%
	\else\ifazwei%
			\SetPaperSize{59.4cm}{42.0cm}   % A2%
		\else\ifadrei%
				\SetPaperSize{42.0cm}{29.7cm}   % A3%
			\else\ifavier%
					\SetPaperSize{29.7cm}{21.0cm}   % A4 example%
				\fi\fi\fi\fi\fi%
%===============================================================================%
% Skip lengths%
%===============================================================================%
\setlength\smallskipamount{6pt plus 2pt minus 2pt}%
\setlength\medskipamount{12pt plus 4pt minus 4pt}%
\setlength\bigskipamount{24pt plus 8pt minus 8pt}%
%
\setlength\abovecaptionskip{25pt}%
\setlength\belowcaptionskip{0pt}%
\setlength\abovedisplayskip{25pt plus 6pt minus 15pt}%
\setlength\abovedisplayshortskip{0pt plus 6pt}%
\setlength\belowdisplayshortskip{13pt plus 7pt minus 6pt}%
\setlength\belowdisplayskip\abovedisplayskip%
%
%===============================================================================%
% Margin setup%
%===============================================================================%
\newlength{\myleftmargin}%
\newlength{\headermargin}%
\newlength{\footermargin}%
\newlength{\bottommargin}%
%
\ifanull%
\setlength{\myleftmargin}{4.6cm}% This is set for right, top, bottom as well
\setlength{\headermargin}{18cm}%
\setlength{\footermargin}{\ifportrait 9cm\else 15cm\fi}%
\else\ifavier%
	\setlength{\myleftmargin}{0.5cm}%
	\setlength{\headermargin}{7cm}%
	\setlength{\headheight}{0.15\paperheight}%
	\setlength{\headsep}{0.001\paperheight}%
	\setlength{\footermargin}{\ifportrait 7cm\else 5cm\fi}%
	\setlength{\footskip}{5pt}%
\else\relax\fi%
%
\setlength{\topmargin}{\dimexpr\myleftmargin+\headermargin\relax}%
\setlength{\bottommargin}{\dimexpr\myleftmargin+\footermargin\relax}%
\setlength{\oddsidemargin}{\dimexpr \myleftmargin - 1in \relax}%
\setlength{\marginparsep}{0cm}%
\setlength{\marginparwidth}{0cm}%
%
% Compute final text dimensions%
\setlength{\textheight}{\dimexpr\paperheight-\topmargin-\bottommargin\relax}%
\setlength{\textwidth}{\dimexpr\paperwidth-2\myleftmargin\relax}%

% Swap dimensions if portrait%
\ifportrait%
	\newdimen\tempdimen%
	\tempdimen=\paperwidth \setlength{\paperwidth}{\paperheight} \setlength{\paperheight}{\tempdimen}%
	\tempdimen=\textwidth \setlength{\textwidth}{\textheight} \setlength{\textheight}{\tempdimen}%
	% Recalculate the textwidth and textheight
	\setlength{\textheight}{\dimexpr\paperheight-\topmargin-\bottommargin\relax}%
	\setlength{\textwidth}{\dimexpr\paperwidth-2\myleftmargin\relax}%
\fi%
%

%
%
%===============================================================================%
% Colors
%===============================================================================%
%scads.ai colors%
\definecolor{scadsaiblack}{HTML}{1a1a1a}%
\definecolor{scadsaitorquise}{HTML}{009299}%
\definecolor{scadsaiblue}{HTML}{0074AC}%
\definecolor{scadsaidarkblue}{HTML}{004B6F}%
\definecolor{scadsaidarkgrey}{HTML}{7F7F7F}%
\definecolor{scadsaigreen}{HTML}{71BD56}%
\definecolor{scadsaigrey}{HTML}{7A7B7A}%
\definecolor{scadsailightgrey}{HTML}{DADADA}%
\definecolor{scadsaimiddleblue}{HTML}{005277}%
\definecolor{scadsaiwhite}{HTML}{ffffff}%
% For coding
\definecolor{codebg}{rgb}{0.95,0.95,0.95}
\definecolor{coderule}{rgb}{0.8,0.8,0.8}
\definecolor{codekw}{rgb}{0.2,0.2,0.7}
\definecolor{codestr}{rgb}{0.65,0.1,0.1}
\definecolor{codecm}{rgb}{0.25,0.5,0.35}
%
%===============================================================================%
% Other%
%===============================================================================%
% Hyperlink
\usepackage[colorlinks=true,urlcolor=scadsaiblue,implicit=false]{hyperref}%
%
% Itemize and enumerate
\renewcommand{\labelitemi}{$\vcenter{\hbox{\textcolor{scadsaiblue}{\small$\blacksquare$}}}$}%
\renewcommand{\labelitemii}{$\vcenter{\hbox{\textcolor{scadsaiblue}{\footnotesize$\blacksquare$}}}$}%
\renewcommand{\labelitemiii}{$\vcenter{\hbox{\textcolor{scadsaiblue}{\tiny$\blacksquare$}}}$}%
\setlist[enumerate]{%
	leftmargin=1.5cm,    % Indentation from left margin%
	% labelindent=0.5cm, % Space between margin and label%
	% itemindent=1cm,    % Indent of wrapped lines%
	% itemsep=0.5\baselineskip, % Vertical space between items%
	parsep=2.5mm       % Vertical space within items%
}%
%
\setlist[itemize]{%
	leftmargin=1.5cm,    % Indentation from left margin%
	% labelindent=0.5cm, % Space between margin and label%
	% itemindent=1cm,    % Indent of wrapped lines%
	% itemsep=0.5\baselineskip, % Vertical space between items%
	parsep=2.5mm       % Vertical space within items%
}%
%
%===============================================================================%
% Parsing input Title, Subtitle, Author%
%===============================================================================%
% Title%
\newcommand\subtitle[1]{\renewcommand\@subtitle{#1}}%
\def\@subtitle{}%
\newcommand\affiliation[1]{\renewcommand\@affiliation{#1}}%
\def\@affiliation{}%
%
% Author under the title%
\newcommand\myauthor[1]{\renewcommand\@myauthor{#1}}%
\def\@myauthor{}%
%
% Author details for the footer%
\newcommand\myfooterauthor[1]{\renewcommand\@myfooterauthor{#1}}%
\def\@myfooterauthor{}%
\newcommand\myfooteremail[1]{\renewcommand\@myfooteremail{#1}}%
\def\@myfooteremail{}%
\newcommand\myfooteruni[1]{\renewcommand\@myfooteruni{#1}}%
\def\@myfooteruni{}%
%
%
%===============================================================================%
% Defining new commands%
%===============================================================================%
\def\tand{&} % '&' for the footer table%
%
\newcommand\mytablecontentsAuthorName{}%
\newcommand\mytablecontentsAuthorEmail{}%
\newcommand\mytablecontentsAuthorUni{}%
%
\newcommand\authornum{}%
\newcommand{\myauthornum}[1]{\renewcommand*{\authornum}{#1}}%
%
% Need 'expl3' and 'xparse' packages%
\ExplSyntaxOn%
\cs_generate_variant:Nn \clist_count:n { V }%
\cs_new_eq:NN \countItems \clist_count:V%
\ExplSyntaxOff%
%
\newlength{\rulewidth}\setlength{\rulewidth}{3pt}%
\newcommand{\spacechar}{\texttt{\char32}}%
%
\newcommand{\additionallogo}[1]{\def\additionallogomacro{#1}}
\def\additionallogomacro{} % Initialize empty
%
%===============================================================================%
% Background design%
%===============================================================================%
\newcommand{\setScadsaiPosterThemeTwo}{%
	\newcommand{\getResearchAreaImage}{}%
	%===============================================================================%
	% Title, Subtitle, Author%
	%===============================================================================%
	\begin{tikzpicture}[remember picture, overlay, inner sep=0mm, outer sep=0mm]%
		% ScaDS.AI Logo%
		\node (scadslogo) [at=(current page.north west), anchor=north west, shift={(\myleftmargin,-\myleftmargin)}]%
		{\includegraphics[height=5cm]{templatepictures/logos/scads-logo.png}};%
		%
		\node[right=5cm of scadslogo.east, anchor=west, align=center]{
			\foreach [count=\i] \x in \additionallogomacro {%
				\includegraphics[height=5cm]{\x}\hspace{5cm}%
			}%
		};
		%
		% Top right design%
		\node (topright) [at=(current page.north east), anchor=north east]%
		{\includegraphics[width=31.686cm]{templatepictures/design/network.png}};%
		%
		%
		% Title%
		\node (title) [%
			at=(current page.north west), anchor=north west,%
			text width=\textwidth, shift={(4.58,-12.3cm)},font=\setmaintitlefont%
		]{\noindent\color{scadsaidarkblue}\@title};%
		%
		% Subtitle%
		\node (subtitle) [below=1.73cm of title.south, anchor=north, text width=\textwidth, yshift=1ex]{%
			\ifx%
				\@subtitle\empty%
				\else%
				{\color{scadsaiblack}\setmaintitlesubtitlefont\strut\@subtitle}%
			\fi%
		};%
		%
		% Author%
		\node (titleAuthor) [below=1.47cm of subtitle.south, anchor=north, text width=\textwidth, font=\setmaintitleauthorfont,yshift=1ex]{%
			\foreach \i [count=\x] in \@myauthor%
			{\i%
				\ifnum\x=\countItems{\@myauthor}%
				\else%
					,\spacechar%
				\fi}%
		};%
		%
		% Author Affiliation%
		\node (titleAuthorAffiliation) [below=1mm of titleAuthor.south, anchor=north, text width=\textwidth, font=\setmaintitleaffiliationfont]{%
			\@affiliation%
		};%
		%
		% Bottom waves%
		\node (bottomWaves) [at=(current page.south), anchor=south, text width=\paperwidth]{%
			\includegraphics[width=\paperwidth]{templatepictures/design/bottom_waves_v2.png}%
		};%
	\end{tikzpicture}%
	% \end{minipage}%
}%
%
\newcommand{\makebackground}{%
	\setScadsaiPosterThemeTwo%
}%
%
%
%
%
%
%===============================================================================%
% Footer%
%===============================================================================%
\newlength{\footerauthorwidtheach}%
\newlength{\footerlogoheight}\setlength{\footerlogoheight}{2.5cm}%
% Footer portrait%
% \newcommand{\footercontactheader}{{\color{scadsaidarkblue}\setfootertitlefont\scadsai@voccontact}} %NR%
\newcommand{\footercontactdetails}{%
	\ifnum\countItems{\@myfooterauthor}=1%
		\setlength{\footerauthorwidtheach}{\linewidth}%
	\else%
		\ifnum\countItems{\@myfooterauthor}=2%
			\setlength{\footerauthorwidtheach}{0.5\linewidth} % max 2 (1/2=0.5) authors can fit in portrait mode%
		\else%
			\setlength{\footerauthorwidtheach}{0.33333333333333333\linewidth} % max 3 (1/3=0.333) authors can fit in portrait mode%
		\fi%
	\fi%
	\begin{flushleft}
		\vspace{-\baselineskip}
		\begin{tabular}[t]{@{}*{\countItems{\@myfooterauthor}}{>{\ignorespaces\raggedright\arraybackslash}p{\footerauthorwidtheach}@{}}}
			\mytablecontentsAuthorUni   \\[5pt]%
			\mytablecontentsAuthorName  \\[5pt]%
			\mytablecontentsAuthorEmail \\%
		\end{tabular}
		\vspace{-\baselineskip}
	\end{flushleft}
}%
%
\newcommand{\makefooter}{%
	\noindent%
	\begin{tikzpicture}[remember picture, overlay, inner sep=0mm, outer sep=0mm]%
		\node (ruler) [at=(current page.north west), anchor=north west,text width=\textwidth, xshift=\myleftmargin, yshift=-\dimexpr\topmargin+\textheight\relax] {\color{scadsaiblue}\rule{\textwidth}{\rulewidth}};%
		%
		% QR code and link%
		\node (qrcode) [below=1.25cm of ruler.north west, anchor=north west] {\includegraphics[width=\dimexpr9.3cm-\myleftmargin\relax]{templatepictures/logos/qr-code.png}};%
		\node (qrcodelink) [below=5mm of qrcode.south, text width=\dimexpr9.3cm-\myleftmargin\relax, font=\setthemainfont\scriptsize] {https://scads.ai};%
		%
		% Contact%
		\node (contactHeader) [right=2cm of qrcode.north east, anchor=north west, font=\setfootertitlefont] {\noindent\color{scadsaidarkblue}\MakeUppercase{\scadsai@voccontact}};%
		%
		\node (contactName) [below=1cm of contactHeader.south west, anchor=north west, font=\setfootertitlefont, text width=0.4\textwidth] {\noindent\footercontactdetails};%
		%
		\node (cooperationHeader) [right=26cm of contactHeader.north east, anchor=north west, font=\setfootertitlefont, text width=0.2\textwidth] {\color{scadsaidarkblue}\MakeUppercase{\scadsai@voccooperation}};%
		%
		%
		\node (cooperationLogo) [below=1.5cm of cooperationHeader.south west, anchor=north west, text width=0.2\textwidth] {%
			\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-ul.png}%
			\hfill%
			\ifgerman%
				\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-tud-de.png}%
			\else%
				\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-tud-en.png}%
			\fi%
		};%
		%
		\node (fundedHeadline) [right=3.5cm of cooperationHeader.north east, anchor=north west, font=\setfootertitlefont, text width=0.23\textwidth] {\color{scadsaidarkblue}\MakeUppercase{\scadsai@vocsponsoredby}};%
		%
		%
		\node (cooperationLogo) [below=1.5cm of fundedHeadline.south west, anchor=north west, text width=0.25\textwidth] {%
			\ifgerman%
				\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-bmftr.png}%
			\else%
				\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-bmftr-en.png}%
			\fi%
			\hfill%
			\includegraphics[height=\footerlogoheight]{templatepictures/logos/logo-sachsen.png}%
			\hspace*{2mm}%
			\includegraphics[height=\footerlogoheight]{templatepictures/logos/Text_Bund_Land.png}%
		};%
		% Acknowledge%
		\node (contactHeader) [below=2mm of qrcodelink.south west, anchor=north west, font=\opensansfont\scriptsize, text width=\textwidth, align=center] {\scadsai@vocacknowledgement};%
	\end{tikzpicture}%
}%
%
\pagestyle{fancy}%
\fancyhf{}%
% Header%
\fancyhead[L]{}%
\renewcommand{\headrulewidth}{0cm}%
%Footer%
\fancyfoot[L]{\makefooter}%
\renewcommand{\footrulewidth}{0cm}%
\newcommand{\parseFooterInput}{
	\foreach \i [count=\xi] in \@myfooterauthor{%
		\gappto\mytablecontentsAuthorName{\setfooterauthorfont}%
		\xappto\mytablecontentsAuthorName{\i}%
		\ifnum\xi=\countItems{\@myfooterauthor}%
		\else%
			\gappto\mytablecontentsAuthorName{\tand}%
		\fi%
	}%
	\foreach \j [count=\xj] in \@myfooteruni{%
		\gappto\mytablecontentsAuthorUni{\color{scadsaidarkblue}\setfooterunivfont}%
		\xappto\mytablecontentsAuthorUni{\j}%
		\ifnum\xj=\countItems{\@myfooteruni}%
		\else%
			\gappto\mytablecontentsAuthorUni{\tand}%
		\fi%
	}%
	\foreach \k [count=\xk] in \@myfooteremail{%
		\gappto\mytablecontentsAuthorEmail{\setfooterauthoremailfont}%
		\xappto\mytablecontentsAuthorEmail{\k}%
		\ifnum\xk=\countItems{\@myfooteremail}%
		\else%
			\gappto\mytablecontentsAuthorEmail{\tand}%
		\fi%
	}%
	%
	\ifcomputetimeused\else\renewcommand*{\scadsai@vocacknowledgement}{}\fi%
}
%
%===============================================================================%
% Section definition: Requrires [explicit]{titlesec}%
%===============================================================================%
\titleformat%
{\section}% command%
[block]% shape%
{\setsectiontitlefont}% format%
{}% label%
{0pt}% separator%
{% before code%
	{%
			\color{scadsaidarkblue}%
			\if@useSectionTitleCenter%
				\hfill\if@useSectionNum\thesection~~\fi\centering{#1}\hfill%
			\else%
				\if@useSectionNum\thesection~~\fi{#1}%
			\fi%
		}%
}%
[% after code%
	\vspace{-1cm}% Space between title and line%
	{\color{scadsaiblue}\noindent\rule{\textwidth}{\rulewidth}}% Line%
	\vspace{2mm}% Space between line and content%
]%
%
% For having numberless reference section title%
\titleformat{name=\section,numberless}[block]{\setsectiontitlefont}{}{0pt}{%
	{\color{scadsaidarkblue}\if@useSectionTitleCenter\hfill\centering{#1}\hfill\else{#1}\fi}%
}[%
	\vspace{-1cm}% Space between title and line%
	{\color{scadsaiblue}\noindent\rule{\textwidth}{\rulewidth}}% Line%
	\vspace{2mm}% Space between line and content%
]%
%
% For general subsection%
\titleformat%
{\subsection}%command%
[block]%shape%
{\setsubsectiontitlefont}%format%
{}%label%
{0pt}%separator%
{% before code%
	{%
			\color{scadsaidarkblue}%
			\if@useSectionTitleCenter%
				\hfill\if@useSectionNum\thesubsection~~\fi\centering{#1}\hfill%
			\else%
				\if@useSectionNum\thesubsection~~\fi{#1}%
			\fi%
		}%
}%
[%after code%
	\vspace*{.4cm}%
]%
%
% For having numberless reference subsection title%
\titleformat{name=\subsection,numberless}[block]{\setsubsectiontitlefont}{}{0pt}{%
	{\color{scadsaidarkblue}\if@useSectionTitleCenter\hfill\centering{#1}\hfill\else{#1}\fi}%
}[\vspace*{.4cm}]%
%
%
%
\AtBeginDocument{%
	\cleanlookdateon % Removes the ordinal numbering of the day
	\setthemainfont%
	\ifenglish\englishvocabulary\else\germanvocabulary\fi% Setting up language%
	\noindent%
	\makebackground%
}%
%
\AtEndDocument{%
	\parseFooterInput%
}%
%
\endinput%
%% %
%% End of file `cls'.%
