import React, { useState } from 'react';
import { Search, Loader2, Lightbulb, ChevronRight } from 'lucide-react';
import { GoogleGenAI, Type } from '@google/genai';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';

let aiInstance: GoogleGenAI | null = null;
const getAI = () => {
  if (!aiInstance) {
    aiInstance = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return aiInstance;
};

interface Meaning {
  meaning: string;
  example: string;
}

interface Syllable {
  char: string;
  isHanja: boolean;
  hanjaChar?: string;
  hanjaMeaning?: string;
  relatedWords?: string[];
}

interface DictionaryResult {
  word: string;
  meanings: Meaning[];
  syllables: Syllable[];
}

export default function App() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<DictionaryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // State for the interactive learning feature
  const [selectedSyllableIndex, setSelectedSyllableIndex] = useState<number | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [showMeanings, setShowMeanings] = useState(false);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);
    setSelectedSyllableIndex(null);
    setShowHint(false);
    setShowMeanings(false);

    try {
      const ai = getAI();
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `다음 낱말을 초등학생이 이해하기 쉽게 설명하고, 각 글자(음절)의 한자 어원을 분석해줘.
1. 뜻과 예문: 초등학생 눈높이에 맞춰 쉽고 명확하게 작성. 뜻이 여러 개면 배열에 모두 넣어줘. 예문에는 해당 낱말을 **굵게** 표시해줘.
2. 음절 분석: 낱말을 한 글자씩 나누어 한자어인지 순우리말/외래어인지 정확히 분석해줘. (예: '촛불'에서 '촛'은 순우리말 '초'에 사이시옷이 붙은 것이므로 한자가 아님. '불'도 순우리말임. 따라서 둘 다 isHanja는 false여야 함)
3. 한자어인 경우: 실제 한자(hanjaChar)와 초등학생이 이해하기 쉬운 한자의 순수한 뜻(hanjaMeaning, 예: '높다', '곧', '오래되다' 등. 한자의 음은 제외하고 뜻만 적어주세요.), 그리고 **반드시 해당 한자가 포함된 초등학생이 알 만한 아주 쉬운 단어** 3~4개를 'relatedWords'에 넣어줘. (주의: 뜻만 비슷한 단어가 아니라, 반드시 그 한자 글자가 들어간 단어여야 해. 또한 한글 표기만 같고 한자가 다른 동음이의어는 절대 넣지 마. 예: '사(辭)'의 relatedWords에 '인사(人事)'를 넣으면 안 됨.)
4. 순우리말이나 외래어인 경우: isHanja를 반드시 false로 설정하고, 나머지 한자 관련 필드는 비워둬.

낱말: ${query.trim()}`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              word: { type: Type.STRING },
              meanings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    meaning: { type: Type.STRING, description: "초등학생 수준의 쉬운 뜻 설명" },
                    example: { type: Type.STRING, description: "해당 낱말이 들어간 짧고 쉬운 예문. 해당 낱말은 **굵게** 표시할 것." },
                  },
                  required: ["meaning", "example"],
                },
              },
              syllables: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    char: { type: Type.STRING, description: "낱말의 한 글자" },
                    isHanja: { type: Type.BOOLEAN, description: "한자어인지 여부 (순우리말/외래어는 false)" },
                    hanjaChar: { type: Type.STRING, description: "실제 한자 (예: '高')" },
                    hanjaMeaning: { type: Type.STRING, description: "초등학생이 이해하기 쉬운 한자의 순수한 뜻 (예: '높다', '곧', '오래되다' 등. 한자의 음은 제외하고 뜻만 적어주세요.)" },
                    relatedWords: { 
                      type: Type.ARRAY, 
                      items: { type: Type.STRING },
                      description: "반드시 해당 한자가 포함된 초등학생 수준의 아주 쉬운 단어들 (한글 표기만 같고 한자가 다른 단어는 절대 불가)"
                    }
                  },
                  required: ["char", "isHanja"],
                }
              }
            },
            required: ["word", "meanings", "syllables"],
          },
        },
      });

      let jsonStr = response.text || '{}';
      // AI가 마크다운(```json ... ```) 형태로 응답할 경우를 대비해 텍스트 정제
      jsonStr = jsonStr.replace(/```json/gi, '').replace(/```/g, '').trim();
      
      const parsedResult = JSON.parse(jsonStr) as DictionaryResult;
      
      if (!parsedResult.word || parsedResult.meanings.length === 0) {
        setError('결과를 찾을 수 없어요.');
      } else {
        setResult(parsedResult);
      }
    } catch (err: any) {
      console.error(err);
      // 에러 메시지를 화면에 표시하여 원인 파악 (할당량 초과, JSON 파싱 에러 등)
      setError(`오류가 발생했어요: ${err?.message || '다시 시도해주세요.'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSyllableClick = (index: number, isHanja: boolean) => {
    if (!isHanja) return;
    if (selectedSyllableIndex === index) {
      setSelectedSyllableIndex(null);
    } else {
      setSelectedSyllableIndex(index);
      setShowHint(false);
    }
  };

  const hasHanja = result?.syllables.some(s => s.isHanja);
  const isInitial = !result && !loading;

  return (
    <div className={`h-screen bg-[#f8fafc] flex flex-col items-center py-6 px-6 font-sans overflow-hidden transition-all duration-700 ${isInitial ? 'justify-center' : 'justify-start'}`}>
      <div className={`w-full max-w-[95vw] 2xl:max-w-[1600px] flex flex-col gap-6 min-h-0 ${isInitial ? '' : 'h-full'}`}>
        <motion.form 
          layout
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, type: "spring", bounce: 0.2 }}
          onSubmit={handleSearch} 
          className={`relative mx-auto w-full shrink-0 transition-all duration-700 ${isInitial ? 'max-w-3xl' : 'max-w-4xl'}`}
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="궁금한 낱말을 적어보세요"
            className={`w-full rounded-full border-4 border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-100 outline-none transition-all duration-700 bg-white font-bold text-slate-800 placeholder:text-slate-400 ${
              isInitial 
                ? 'pl-10 pr-24 py-6 text-4xl shadow-2xl' 
                : 'pl-8 pr-20 py-4 text-2xl shadow-md'
            }`}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className={`absolute top-1/2 -translate-y-1/2 text-slate-400 hover:text-blue-500 disabled:opacity-50 transition-colors rounded-full hover:bg-slate-50 ${
              isInitial ? 'right-4 p-4' : 'right-3 p-3'
            }`}
          >
            {loading ? (
              <Loader2 className={`animate-spin text-blue-500 ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            ) : (
              <Search className={`transition-all duration-700 ${isInitial ? 'w-10 h-10' : 'w-8 h-8'}`} />
            )}
          </button>
        </motion.form>

        <AnimatePresence mode="wait">
          {error && (
            <motion.div 
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 bg-red-50 text-red-600 rounded-2xl text-center font-bold text-xl max-w-4xl mx-auto w-full shrink-0"
            >
              {error}
            </motion.div>
          )}

          {result && !loading && !error && (
            <motion.div 
              layout
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex ${hasHanja && showMeanings ? 'flex-row max-w-full' : 'flex-col max-w-4xl'} mx-auto gap-6 items-stretch w-full flex-1 min-h-0 pb-2`}
            >
              {/* 1. 확장 기능: 음절 기반 학습 */}
              {hasHanja && (
                <motion.div layout className="w-full p-8 bg-white rounded-[2rem] shadow-sm border-2 border-slate-200/60 flex flex-col flex-1 min-h-0 overflow-y-auto custom-scrollbar">
                  <div className="flex flex-wrap items-center gap-4 mb-6 justify-center shrink-0">
                    {result.syllables.map((syllable, index) => (
                      <React.Fragment key={index}>
                        <button
                          onClick={() => handleSyllableClick(index, syllable.isHanja)}
                          disabled={!syllable.isHanja}
                          className={`
                            w-24 h-24 text-5xl font-black rounded-[1.5rem] flex items-center justify-center transition-all
                            ${!syllable.isHanja 
                              ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                              : selectedSyllableIndex === index
                                ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30 scale-110 ring-4 ring-blue-200'
                                : 'bg-blue-50 text-blue-600 hover:bg-blue-100 hover:scale-105 cursor-pointer shadow-sm border-2 border-blue-200'
                            }
                          `}
                        >
                          {syllable.char}
                        </button>
                        {index < result.syllables.length - 1 && (
                          <span className="text-slate-300 font-black text-4xl">+</span>
                        )}
                      </React.Fragment>
                    ))}
                  </div>

                  {/* 단어 묶음 및 힌트 영역 */}
                  <AnimatePresence mode="wait">
                    {selectedSyllableIndex !== null && result.syllables[selectedSyllableIndex] && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="overflow-hidden mt-auto shrink-0"
                      >
                        <div className="bg-amber-50/80 border-2 border-amber-200 rounded-[2rem] p-6 flex flex-col items-center text-center">
                          <div className="flex flex-wrap justify-center gap-4 mb-6">
                            {result.word && (
                              <span className="px-6 py-3 bg-white rounded-2xl font-black text-2xl text-blue-600 shadow-sm border-4 border-blue-100">
                                {result.word}
                              </span>
                            )}
                            {result.syllables[selectedSyllableIndex].relatedWords?.map((word, i) => (
                              <span key={i} className="px-6 py-3 bg-white rounded-2xl font-bold text-2xl text-slate-700 shadow-sm border-2 border-slate-200 hover:border-amber-300 transition-colors">
                                {word}
                              </span>
                            ))}
                          </div>
                          
                          <div className="w-full h-1 bg-amber-200/50 mb-6 rounded-full" />

                          <p className="text-2xl font-bold text-amber-800 mb-6">
                            👉 과연 '{result.syllables[selectedSyllableIndex].char}' 자의 공통된 뜻은 무엇일까요?
                          </p>

                          {!showHint ? (
                            <button
                              onClick={() => setShowHint(true)}
                              className="px-8 py-4 bg-amber-400 hover:bg-amber-500 text-white font-black text-2xl rounded-full transition-all shadow-md hover:shadow-lg flex items-center gap-2 transform hover:-translate-y-1"
                            >
                              정답 확인하기 <ChevronRight className="w-6 h-6" />
                            </button>
                          ) : (
                            <motion.div 
                              initial={{ opacity: 0, scale: 0.9, y: 10 }}
                              animate={{ opacity: 1, scale: 1, y: 0 }}
                              className="px-10 py-6 bg-blue-500 text-white font-black text-4xl rounded-[2rem] shadow-lg border-4 border-blue-400"
                            >
                              "{result.syllables[selectedSyllableIndex].hanjaMeaning}"
                            </motion.div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {!showMeanings && (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="mt-8 flex flex-col items-center justify-center border-t-2 border-slate-100 pt-6 shrink-0"
                    >
                      <button
                        onClick={() => setShowMeanings(true)}
                        className="px-10 py-5 bg-slate-800 hover:bg-slate-700 text-white font-black text-2xl rounded-full transition-all shadow-lg hover:shadow-xl flex items-center gap-2 transform hover:-translate-y-1"
                      >
                        📖 뜻풀이 확인하기 <ChevronRight className="w-6 h-6" />
                      </button>
                    </motion.div>
                  )}
                </motion.div>
              )}

              {/* 2. 기본 기능: 사전 (뜻 + 예문) */}
              <AnimatePresence>
                {(!hasHanja || showMeanings) && (
                  <motion.div 
                    layout
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="w-full p-8 bg-white rounded-[2rem] shadow-sm border-2 border-slate-200/60 flex flex-col flex-1 min-h-0 overflow-y-auto custom-scrollbar"
                  >
                    {!hasHanja && (
                      <div className="mb-6 p-5 bg-blue-50 text-blue-700 rounded-2xl font-bold text-xl flex items-center gap-3 shrink-0">
                        <span>✨</span> 이 단어는 순우리말(또는 외래어)이라서 한자 비밀이 없어요!
                      </div>
                    )}
                    <div className="space-y-6">
                      {result.meanings.map((item, index) => (
                        <div key={index} className="bg-slate-50 rounded-[1.5rem] border-2 border-slate-200 overflow-hidden shrink-0">
                          {/* 뜻 영역 */}
                          <div className="p-6 bg-white border-b-2 border-slate-100 flex gap-4 items-start">
                            <span className="flex-shrink-0 w-10 h-10 bg-slate-800 text-white rounded-xl flex items-center justify-center font-black text-xl shadow-md">
                              {index + 1}
                            </span>
                            <p className="text-slate-800 font-bold text-2xl leading-snug pt-1">
                              {item.meaning}
                            </p>
                          </div>
                          
                          {/* 예문 영역 */}
                          <div className="p-6 bg-slate-50 flex gap-4 items-start">
                            <span className="flex-shrink-0 w-10 h-10 bg-blue-100 text-blue-600 rounded-xl flex items-center justify-center text-2xl shadow-sm">
                              💡
                            </span>
                            <div className="text-slate-600 text-xl leading-snug pt-1 markdown-body-inline break-keep">
                              <Markdown>{item.example}</Markdown>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
